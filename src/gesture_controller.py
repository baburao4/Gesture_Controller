import pyautogui
import math
import time
import cv2
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


class GestureController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.screen_w, self.screen_h = pyautogui.size()

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        # Smoothing
        self.alpha_fast = 0.50
        self.alpha_slow = 0.30
        self.speed_threshold = 25

        # ── Camera active zone ───────────────────────────────────────────
        # These define the comfortable region of the camera your hand moves in.
        # We then OVER-SCALE the output so reaching ~85% of this zone already
        # maps to 100% of the screen — corners are always reachable.
        self.cam_x_margin = 0.12
        self.cam_y_top    = 0.08
        self.cam_y_bottom = 0.88

        # ── Volume (pycaw) ──────────────────────────────────────────────
        speakers = AudioUtilities.GetSpeakers()
        com_endpoint = getattr(speakers, '_dev', speakers)
        interface = com_endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
        vol_range = self.volume_ctrl.GetVolumeRange()
        self.vol_min, self.vol_max = vol_range[0], vol_range[1]

        # ── Click state machine (thumb + index pinch) ────────────────────
        self.click_state      = "idle"
        self.click_start_time = 0
        self.locked_x         = None
        self.locked_y         = None

        # Raised thresholds so angled/corner pinches still register
        self.PINCH_CLOSE = 45
        self.PINCH_OPEN  = 65
        self.PINCH_HOLD  = 0.06

        # Right-click: middle + thumb pinch (thumb up + middle up only)
        self.rclick_state      = "idle"
        self.rclick_start_time = 0
        self.RCLICK_HOLD       = 0.20

        # Double-click: stamp time when pinch OPENS.
        # If the NEXT pinch starts within DOUBLE_CLICK_GAP, fire doubleClick.
        self.pinch_open_time  = 0
        self.DOUBLE_CLICK_GAP = 0.50

        # ── Drag state machine (open palm) ──────────────────────────────
        self.drag_state      = "idle"
        self.drag_start_time = 0
        self.is_dragging     = False
        self.DRAG_HOLD       = 0.50

        # ── Scroll state ────────────────────────────────────────────────
        # Float accumulator so slow movements still register over time.
        self.scroll_prev_y   = None
        self.scroll_accum    = 0.0
        self.SCROLL_DEADZONE = 4
        self.SCROLL_SPEED    = 0.25

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _map_to_screen(self, x, y, cam_w, cam_h):
        """
        Map camera coords → screen coords with 15% overshoot so the
        inner active zone always covers the full screen including corners.
        """
        x_min = int(cam_w * self.cam_x_margin)
        x_max = int(cam_w * (1 - self.cam_x_margin))
        y_min = int(cam_h * self.cam_y_top)
        y_max = int(cam_h * self.cam_y_bottom)

        overshoot = 0.15
        sx = np.interp(x, (x_min, x_max),
                       (-self.screen_w * overshoot, self.screen_w * (1 + overshoot)))
        sy = np.interp(y, (y_min, y_max),
                       (-self.screen_h * overshoot, self.screen_h * (1 + overshoot)))
        sx = np.clip(sx, 0, self.screen_w - 1)
        sy = np.clip(sy, 0, self.screen_h - 1)
        return sx, sy

    def _finger_states(self, lm_list):
        tips   = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
        joints = {'thumb': 3, 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}
        up = {}
        up['thumb'] = lm_list[tips['thumb']][1] < lm_list[joints['thumb']][1]
        for f in ('index', 'middle', 'ring', 'pinky'):
            up[f] = lm_list[tips[f]][2] < lm_list[joints[f]][2]
        return up

    def _dist(self, lm_list, id_a, id_b):
        return math.hypot(
            lm_list[id_a][1] - lm_list[id_b][1],
            lm_list[id_a][2] - lm_list[id_b][2]
        )

    def _smooth_cursor(self, raw_x, raw_y):
        dx = raw_x - self.prev_x
        dy = raw_y - self.prev_y
        alpha = self.alpha_fast if math.hypot(dx, dy) > self.speed_threshold else self.alpha_slow
        new_x = alpha * raw_x + (1 - alpha) * self.prev_x
        new_y = alpha * raw_y + (1 - alpha) * self.prev_y
        self.prev_x, self.prev_y = new_x, new_y
        return new_x, new_y

    def _release_drag(self):
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
        self.drag_state = "idle"

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 1 — Cursor  (index only)
    # ────────────────────────────────────────────────────────────────────

    def _handle_cursor(self, lm_list, img):
        cam_h, cam_w, _ = img.shape
        index = lm_list[8]
        sx, sy = self._map_to_screen(index[1], index[2], cam_w, cam_h)
        new_x, new_y = self._smooth_cursor(sx, sy)
        pyautogui.moveTo(int(new_x), int(new_y))

        cv2.circle(img, (index[1], index[2]), 10, (0, 200, 255), cv2.FILLED)
        cv2.putText(img, "CURSOR", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 2 — Click / Double-Click  (thumb + index pinch)
    #
    # Double-click timing is measured from when the FIRST pinch OPENS.
    # So: pinch → click fires → open fingers (timer starts) → pinch again
    # within 0.5s → doubleClick fires. Natural and reliable.
    # ────────────────────────────────────────────────────────────────────

    def _handle_click(self, lm_list, img):
        cam_h, cam_w, _ = img.shape
        index    = lm_list[8]
        thumb    = lm_list[4]
        distance = self._dist(lm_list, 8, 4)
        mid_x    = (index[1] + thumb[1]) // 2
        mid_y    = (index[2] + thumb[2]) // 2
        now      = time.time()

        sx, sy = self._map_to_screen(index[1], index[2], cam_w, cam_h)

        if self.click_state == "idle":
            new_x, new_y = self._smooth_cursor(sx, sy)
            pyautogui.moveTo(int(new_x), int(new_y))  # always move, no finger-up guard

            if distance < self.PINCH_CLOSE:
                self.click_state      = "pre_click"
                self.click_start_time = now
                self.locked_x = self.prev_x
                self.locked_y = self.prev_y

        elif self.click_state == "pre_click":
            if distance >= self.PINCH_OPEN:
                # Released too fast — cancel pinch, stamp open time
                self.pinch_open_time = now
                self.click_state = "idle"

            elif now - self.click_start_time >= self.PINCH_HOLD:
                # Check if this is a second pinch within the double-click window
                if now - self.pinch_open_time < self.DOUBLE_CLICK_GAP:
                    pyautogui.doubleClick(int(self.locked_x), int(self.locked_y))
                    self.click_state = "double_clicked"
                else:
                    pyautogui.click(int(self.locked_x), int(self.locked_y))
                    self.click_state = "clicked"

        elif self.click_state == "clicked":
            if distance >= self.PINCH_OPEN:
                # Stamp open time HERE so the next pinch can be a double-click
                self.pinch_open_time = now
                self.click_state = "idle"

        elif self.click_state == "double_clicked":
            if distance >= self.PINCH_OPEN:
                self.pinch_open_time = now
                self.click_state = "idle"

        state_colors = {
            "idle":           (255, 0, 255),
            "pre_click":      (0, 165, 255),
            "clicked":        (0, 0, 255),
            "double_clicked": (0, 255, 0),
        }
        color = state_colors[self.click_state]

        cv2.circle(img, (index[1], index[2]), 8, color, cv2.FILLED)
        cv2.circle(img, (thumb[1],  thumb[2]), 8, color, cv2.FILLED)
        cv2.line(img, (index[1], index[2]), (thumb[1], thumb[2]), color, 2)
        cv2.circle(img, (mid_x, mid_y), 6, color, cv2.FILLED)

        gap = now - self.pinch_open_time
        labels = {
            "idle":           f"CLICK  (gap {gap:.2f}s)",
            "pre_click":      f"Pinch {min(int((now - self.click_start_time)/self.PINCH_HOLD*100),99)}%",
            "clicked":        "CLICKED!  — pinch again to double-click",
            "double_clicked": "DOUBLE CLICK! ✓",
        }
        cv2.putText(img, labels[self.click_state], (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(img, f"Dist: {int(distance)}px", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 3 — Scroll  (index + middle up, no thumb)
    #
    # Float accumulator: small per-frame deltas add up before firing,
    # so slow hand movement still produces continuous scroll output.
    # ────────────────────────────────────────────────────────────────────

    def _handle_scroll(self, lm_list, img):
        index  = lm_list[8]
        middle = lm_list[12]
        avg_y  = (index[2] + middle[2]) // 2
        direction = ""

        if self.scroll_prev_y is not None:
            delta = self.scroll_prev_y - avg_y   # +ve = hand up = scroll up

            if abs(delta) > self.SCROLL_DEADZONE:
                self.scroll_accum += delta * self.SCROLL_SPEED
                scroll_units = int(self.scroll_accum)
                if scroll_units != 0:
                    pyautogui.scroll(scroll_units)
                    self.scroll_accum -= scroll_units
                    direction = "▲" if scroll_units > 0 else "▼"
            else:
                self.scroll_accum *= 0.8   # decay when hand is still

        self.scroll_prev_y = avg_y

        mid_x = (index[1] + middle[1]) // 2
        mid_y = (index[2] + middle[2]) // 2
        cv2.circle(img, (index[1],  index[2]),  9, (255, 200, 0), cv2.FILLED)
        cv2.circle(img, (middle[1], middle[2]), 9, (255, 200, 0), cv2.FILLED)
        cv2.line(img,   (index[1],  index[2]), (middle[1], middle[2]), (255, 200, 0), 2)
        cv2.circle(img, (mid_x, mid_y), 6, (255, 200, 0), cv2.FILLED)
        cv2.putText(img, f"SCROLL {direction}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2)

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 4 — Volume  (thumb + middle spread)
    # ────────────────────────────────────────────────────────────────────

    def _handle_volume(self, lm_list, img):
        cam_h, cam_w, _ = img.shape
        thumb  = lm_list[4]
        middle = lm_list[12]
        dist   = self._dist(lm_list, 4, 12)

        vol_db = float(np.clip(
            np.interp(dist, [30, 180], [self.vol_min, self.vol_max]),
            self.vol_min, self.vol_max
        ))
        self.volume_ctrl.SetMasterVolumeLevel(vol_db, None)
        vol_pct = int(np.interp(dist, [30, 180], [0, 100]))

        mid_x = (thumb[1] + middle[1]) // 2
        mid_y = (thumb[2] + middle[2]) // 2

        cv2.circle(img, (thumb[1],  thumb[2]),  9, (0, 255, 128), cv2.FILLED)
        cv2.circle(img, (middle[1], middle[2]), 9, (0, 255, 128), cv2.FILLED)
        cv2.line(img,   (thumb[1],  thumb[2]), (middle[1], middle[2]), (0, 255, 128), 2)
        cv2.circle(img, (mid_x, mid_y), 6, (0, 255, 128), cv2.FILLED)
        cv2.putText(img, f"VOL {vol_pct}%", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2)

        bx, b_top, b_bot = cam_w - 45, 80, 300
        fill_y = int(np.interp(vol_pct, [0, 100], [b_bot, b_top]))
        cv2.rectangle(img, (bx, b_top),  (bx + 18, b_bot),  (50, 50, 50),  cv2.FILLED)
        cv2.rectangle(img, (bx, fill_y), (bx + 18, b_bot),  (0, 255, 128), cv2.FILLED)
        cv2.putText(img, f"{vol_pct}%", (bx - 8, b_bot + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 5 — Drag & Drop  (open palm = all 5 up)
    #
    # DROP GESTURE (NEW):
    #   While dragging, fold ONLY the pinky (keep all others up).
    #   This is deliberate and hard to do accidentally during movement.
    #   Detected in handle_gestures() before the all_up check.
    # ────────────────────────────────────────────────────────────────────

    def _handle_drag(self, lm_list, img):
        cam_h, cam_w, _ = img.shape
        palm = lm_list[9]
        sx, sy   = self._map_to_screen(palm[1], palm[2], cam_w, cam_h)
        new_x, new_y = self._smooth_cursor(sx, sy)
        now = time.time()

        if self.drag_state == "idle":
            self.drag_state      = "arming"
            self.drag_start_time = now
            self.locked_x = self.prev_x
            self.locked_y = self.prev_y

        elif self.drag_state == "arming":
            if now - self.drag_start_time >= self.DRAG_HOLD:
                self.drag_state  = "dragging"
                self.is_dragging = True
                pyautogui.mouseDown(int(self.locked_x), int(self.locked_y))

        elif self.drag_state == "dragging":
            pyautogui.moveTo(int(new_x), int(new_y))
            self.prev_x, self.prev_y = new_x, new_y

        charge_pct = 0
        if self.drag_state == "arming":
            charge_pct = min(int((now - self.drag_start_time) / self.DRAG_HOLD * 100), 99)

        drag_colors = {
            "idle":     (200, 200, 255),
            "arming":   (0, 165, 255),
            "dragging": (0, 0, 255),
        }
        color = drag_colors[self.drag_state]

        for tip_id in [4, 8, 12, 16, 20]:
            cv2.circle(img, (lm_list[tip_id][1], lm_list[tip_id][2]), 7, color, cv2.FILLED)

        drag_labels = {
            "idle":     "PALM detected...",
            "arming":   f"Arming {charge_pct}% — keep still",
            "dragging": "DRAGGING  — fold PINKY to drop",
        }
        cv2.putText(img, drag_labels[self.drag_state], (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 6 — Right-Click  (thumb + middle pinch, index down)
    # Hold the pinch for 0.2s to fire — avoids accidental triggers.
    # ────────────────────────────────────────────────────────────────────

    def _handle_rclick(self, lm_list, img):
        cam_h, cam_w, _ = img.shape
        thumb    = lm_list[4]
        middle   = lm_list[12]
        distance = self._dist(lm_list, 4, 12)
        mid_x    = (thumb[1] + middle[1]) // 2
        mid_y    = (thumb[2] + middle[2]) // 2
        now      = time.time()

        sx, sy = self._map_to_screen(lm_list[8][1], lm_list[8][2], cam_w, cam_h)

        if self.rclick_state == "idle":
            new_x, new_y = self._smooth_cursor(sx, sy)
            pyautogui.moveTo(int(new_x), int(new_y))
            if distance < self.PINCH_CLOSE:
                self.rclick_state      = "pre_rclick"
                self.rclick_start_time = now
                self.locked_x = self.prev_x
                self.locked_y = self.prev_y

        elif self.rclick_state == "pre_rclick":
            if distance >= self.PINCH_OPEN:
                self.rclick_state = "idle"
            elif now - self.rclick_start_time >= self.RCLICK_HOLD:
                pyautogui.rightClick(int(self.locked_x), int(self.locked_y))
                self.rclick_state = "rclicked"

        elif self.rclick_state == "rclicked":
            if distance >= self.PINCH_OPEN:
                self.rclick_state = "idle"

        charge = min(int((now - self.rclick_start_time) / self.RCLICK_HOLD * 100), 100)
        color  = (0, 100, 255) if self.rclick_state == "idle" else \
                 (0, 50,  200) if self.rclick_state == "pre_rclick" else (0, 0, 180)

        cv2.circle(img, (thumb[1],  thumb[2]),  8, color, cv2.FILLED)
        cv2.circle(img, (middle[1], middle[2]), 8, color, cv2.FILLED)
        cv2.line(img,   (thumb[1],  thumb[2]), (middle[1], middle[2]), color, 2)

        label = "RIGHT-CLICK" if self.rclick_state == "rclicked" else \
                f"R-Click {charge}%" if self.rclick_state == "pre_rclick" else "RIGHT-CLICK mode"
        cv2.putText(img, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 7 — Screenshot  (pinky + ring up, others down)
    # Hold for 0.8s to avoid accidental triggers. Saves to Desktop.
    # ────────────────────────────────────────────────────────────────────

    def _handle_screenshot(self, lm_list, img):
        now = time.time()
        if not hasattr(self, '_ss_state'):
            self._ss_state     = "idle"
            self._ss_start     = 0
            self._ss_flash     = 0

        if self._ss_state == "idle":
            self._ss_state = "charging"
            self._ss_start = now

        elif self._ss_state == "charging":
            charge = min(int((now - self._ss_start) / 0.8 * 100), 100)
            cv2.putText(img, f"SCREENSHOT {charge}% — hold...", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 80, 200), 2)
            if now - self._ss_start >= 0.8:
                pyautogui.screenshot(
                    f"C:/Users/{__import__('os').getlogin()}/Desktop/"
                    f"screenshot_{int(now)}.png"
                )
                self._ss_state = "done"
                self._ss_flash = now

        elif self._ss_state == "done":
            cv2.putText(img, "SCREENSHOT SAVED! ✓", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 80, 200), 2)
            # White flash overlay
            if now - self._ss_flash < 0.15:
                overlay = img.copy()
                cv2.rectangle(overlay, (0,0), (img.shape[1], img.shape[0]),
                              (255,255,255), -1)
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            if now - self._ss_flash > 1.0:
                self._ss_state = "idle"

    # ────────────────────────────────────────────────────────────────────
    # GESTURE 8 — Media Controls  (index + pinky up = "rock on" 🤘)
    # Swipe hand LEFT = previous track, RIGHT = next track
    # Hold still = play/pause
    # ────────────────────────────────────────────────────────────────────

    def _handle_media(self, lm_list, img):
        now = time.time()
        if not hasattr(self, '_media_prev_x'):
            self._media_prev_x  = None
            self._media_state   = "idle"   # idle / swiped / held
            self._media_hold_t  = now
            self._media_label   = ""
            self._media_label_t = 0

        index_x = lm_list[8][1]

        if self._media_prev_x is None:
            self._media_prev_x = index_x
            self._media_hold_t = now

        swipe_delta = index_x - self._media_prev_x

        # Play/pause: hand held still for 0.6s
        if abs(swipe_delta) < 15:
            if now - self._media_hold_t >= 0.6 and self._media_state == "idle":
                pyautogui.press('playpause')
                self._media_state   = "held"
                self._media_label   = "⏯  PLAY / PAUSE"
                self._media_label_t = now
        else:
            self._media_hold_t = now
            self._media_state  = "idle"

        # Swipe right = next, swipe left = prev (threshold 80px)
        if self._media_state == "idle":
            if swipe_delta > 80:
                pyautogui.press('nexttrack')
                self._media_state   = "swiped"
                self._media_label   = "⏭  NEXT TRACK"
                self._media_label_t = now
            elif swipe_delta < -80:
                pyautogui.press('prevtrack')
                self._media_state   = "swiped"
                self._media_label   = "⏮  PREV TRACK"
                self._media_label_t = now

        if self._media_state in ("swiped", "held") and now - self._media_label_t > 0.8:
            self._media_state  = "idle"
            self._media_prev_x = None

        self._media_prev_x = index_x

        label_color = (255, 180, 0)
        cv2.circle(img, (lm_list[8][1],  lm_list[8][2]),  9, label_color, cv2.FILLED)
        cv2.circle(img, (lm_list[20][1], lm_list[20][2]), 9, label_color, cv2.FILLED)
        display = self._media_label if self._media_label else "MEDIA — swipe or hold"
        cv2.putText(img, display, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)

    # ────────────────────────────────────────────────────────────────────
    # Main entry — gesture router
    # ────────────────────────────────────────────────────────────────────

    def handle_gestures(self, lm_list, img):
        cam_h, cam_w, _ = img.shape
        up = self._finger_states(lm_list)

        # Active-zone overlay
        cv2.rectangle(img,
            (int(cam_w * self.cam_x_margin),      int(cam_h * self.cam_y_top)),
            (int(cam_w * (1 - self.cam_x_margin)), int(cam_h * self.cam_y_bottom)),
            (0, 200, 255), 2)

        # ── Classify ────────────────────────────────────────────────────
        all_up      = all(up[f] for f in ('thumb','index','middle','ring','pinky'))

        # Drop signal: all fingers up EXCEPT pinky
        drop_signal = (up['thumb'] and up['index'] and up['middle']
                       and up['ring'] and not up['pinky'])

        vol_mode     = (    up['thumb'] and not up['index'] and     up['middle']
                        and not up['ring'] and not up['pinky'])
        scroll_mode  = (not up['thumb'] and     up['index'] and     up['middle']
                        and not up['ring'] and not up['pinky'])
        click_mode   = (    up['thumb'] and     up['index']
                        and not up['middle'] and not up['ring'] and not up['pinky'])
        cursor_mode  = (not up['thumb'] and     up['index']
                        and not up['middle'] and not up['ring'] and not up['pinky'])
        rclick_mode  = (    up['thumb'] and not up['index'] and not up['middle']
                        and not up['ring'] and not up['pinky'])   # thumb only
        # 🤘 rock-on: index + pinky up, middle+ring+thumb down
        media_mode   = (not up['thumb'] and     up['index'] and not up['middle']
                        and not up['ring'] and     up['pinky'])
        # 📸 screenshot: ring + pinky up, others down
        screenshot_mode = (not up['thumb'] and not up['index'] and not up['middle']
                           and up['ring'] and up['pinky'])

        # ── Dispatch ────────────────────────────────────────────────────

        # Drop check FIRST — catches pinky-fold while dragging
        if self.is_dragging and drop_signal:
            self._release_drag()
            self.scroll_prev_y = None
            cv2.putText(img, "DROPPED!", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            mode, mc = "DROP", (0, 255, 0)

        elif all_up:
            self.scroll_prev_y = None
            self._handle_drag(lm_list, img)
            mode, mc = "DRAG", (0, 0, 255)

        elif vol_mode:
            self._release_drag()
            self.scroll_prev_y = None
            self._handle_volume(lm_list, img)
            mode, mc = "VOLUME", (0, 255, 128)

        elif scroll_mode:
            self._release_drag()
            self._handle_scroll(lm_list, img)
            mode, mc = "SCROLL", (255, 200, 0)

        elif click_mode:
            self._release_drag()
            self.scroll_prev_y = None
            self._handle_click(lm_list, img)
            mode, mc = "CLICK", (255, 0, 255)

        elif cursor_mode:
            self._release_drag()
            self.scroll_prev_y = None
            self._handle_cursor(lm_list, img)
            mode, mc = "CURSOR", (0, 200, 255)

        elif rclick_mode:
            self._release_drag()
            self.scroll_prev_y = None
            self._handle_rclick(lm_list, img)
            mode, mc = "R-CLICK", (0, 100, 255)

        elif media_mode:
            self._release_drag()
            self.scroll_prev_y = None
            self._handle_media(lm_list, img)
            mode, mc = "MEDIA", (255, 180, 0)

        elif screenshot_mode:
            self._release_drag()
            self.scroll_prev_y = None
            self._handle_screenshot(lm_list, img)
            mode, mc = "SCREENSHOT", (255, 80, 200)

        else:
            self._release_drag()
            self.scroll_prev_y = None
            mode, mc = "—", (100, 100, 100)

        cv2.putText(img, f"Mode: {mode}", (10, cam_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mc, 2)

        # ── Gesture cheat-sheet — bottom right corner ────────────────────
        hints = [
            ("☝  Index         ", "CURSOR"),
            ("👍+☝  Thumb+Index ", "CLICK / DBL"),
            ("👍      Thumb only", "RIGHT-CLICK"),
            ("☝+✌  Idx+Mid     ", "SCROLL"),
            ("👍+✌  Thm+Mid    ", "VOLUME"),
            ("🖐  All 5         ", "DRAG"),
            ("🤘  Idx+Pinky     ", "MEDIA"),
            ("💅  Ring+Pinky    ", "SCREENSHOT"),
        ]
        line_h = 18
        start_y = cam_h - (len(hints) * line_h) - 10
        for i, (gesture, action) in enumerate(hints):
            y = start_y + i * line_h
            active = action.split("/")[0].strip() in mode
            color  = (0, 255, 255) if active else (130, 130, 130)
            cv2.putText(img, f"{gesture}→ {action}", (cam_w - 310, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)