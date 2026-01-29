# import matplotlib.pyplot as plt
# import numpy as np
# from pynput import keyboard


# class JointPlotter:
#     def __init__(self, joint_names, history_len=200, render_stride=5):
#         """
#         joint_names: list[str]  (左右对称 joint 的“基名”，如 hip_pitch)
#         history_len: 曲线窗口长度
#         render_stride: 每多少 step 才 render 一次（防止卡）
#         """
#         self.joint_names = joint_names
#         self.n = len(joint_names)
#         self.history_len = history_len
#         self.render_stride = render_stride

#         self.step = 0
#         self.enable_render = False

#         # ===== 跨线程 flag（pynput → 主线程）=====
#         self.request_close = False
#         self.request_force_render = False
#         self._closed = False

#         # ===== matplotlib init =====
#         plt.ion()
#         self.fig, self.axs = plt.subplots(self.n, 3, figsize=(18, 3 * self.n))

#         # ===== 数据缓存 =====
#         self.data = {
#             "torque": [ {"L": [], "R": []} for _ in range(self.n) ],
#             "pos":    [ {"L": [], "R": []} for _ in range(self.n) ],
#             "vel":    [ {"L": [], "R": []} for _ in range(self.n) ],
#         }

#         # ===== 曲线句柄 =====
#         self.lines = {
#             "torque": [],
#             "pos": [],
#             "vel": [],
#         }

#         for i, name in enumerate(joint_names):
#             for j, key in enumerate(["torque", "pos", "vel"]):
#                 ax = self.axs[i, j]

#                 line_l, = ax.plot([], [], lw=1.5, label="L")
#                 line_r, = ax.plot([], [], lw=1.5, label="R")

#                 self.lines[key].append((line_l, line_r))

#                 ax.set_title(f"{name} | {key}")
#                 ax.grid(True)
#                 ax.legend()

#         plt.tight_layout()
#         plt.show(block=False)
#         self.fig.canvas.draw()
#         try:
#             self.fig.canvas.manager.window.attributes("-topmost", 0)
#         except Exception:
#             pass

#         # ===== 启动全局键盘监听 =====
#         self._start_keyboard_listener()

#         print(
#             "[JointPlotter] ready | keys: "
#             "p(toggle), l(refresh), q(quit)"
#         )

#     # ------------------------------------------------------------------
#     # 键盘监听（子线程）
#     # ------------------------------------------------------------------
#     def _start_keyboard_listener(self):
#         def on_press(key):
#             try:
#                 if key.char == "p":
#                     self.enable_render = not self.enable_render
#                     print(f"[Plot] render = {self.enable_render}")

#                 elif key.char == "l":
#                     print("[Plot] force refresh")
#                     self.request_force_render = True

#                 elif key.char == "q":
#                     print("[Plot] request close")
#                     self.request_close = True
#                     return False
#             except AttributeError:
#                 pass

#         self._listener = keyboard.Listener(on_press=on_press)
#         self._listener.daemon = True
#         self._listener.start()

#     # ------------------------------------------------------------------
#     # 主更新接口（必须在主线程调用）
#     # ------------------------------------------------------------------
#     def update(self, tau_l, tau_r, pos_l, pos_r, vel_l, vel_r):
#         if self._closed:
#             return

#         self.step += 1

#         # ===== 数据更新（轻量）=====
#         for i in range(self.n):
#             self.data["torque"][i]["L"].append(tau_l[i])
#             self.data["torque"][i]["R"].append(tau_r[i])

#             self.data["pos"][i]["L"].append(pos_l[i])
#             self.data["pos"][i]["R"].append(pos_r[i])

#             self.data["vel"][i]["L"].append(vel_l[i])
#             self.data["vel"][i]["R"].append(vel_r[i])

#         # ===== GUI 操作（只在主线程）=====
#         if self.request_force_render:
#             self._render()
#             self.request_force_render = False

#         if self.enable_render and self.step % self.render_stride == 0:
#             self._render()

#         if self.request_close:
#             self._do_close()
#             return

#         # 让 matplotlib 处理事件（不抢焦点）
#         if self.step % self.render_stride == 0:
#             plt.pause(0.001)

#     # ------------------------------------------------------------------
#     # 真正绘图
#     # ------------------------------------------------------------------
#     def _render(self):
#         x = np.arange(
#             max(0, self.step - self.history_len),
#             self.step
#         )

#         for i in range(self.n):
#             for key in ["torque", "pos", "vel"]:
#                 yL = self.data[key][i]["L"][-self.history_len:]
#                 yR = self.data[key][i]["R"][-self.history_len:]

#                 line_l, line_r = self.lines[key][i]
#                 line_l.set_data(x, yL)
#                 line_r.set_data(x, yR)

#                 ax = line_l.axes
#                 if len(x) > 1:
#                     ax.set_xlim(x[0], x[-1])
#                 ax.relim()
#                 ax.autoscale_view()

#         self.fig.canvas.draw_idle()

#     # ------------------------------------------------------------------
#     # 安全关闭（主线程）
#     # ------------------------------------------------------------------
#     def _do_close(self):
#         if self._closed:
#             return
#         self._closed = True
#         plt.close(self.fig)
#         print("[JointPlotter] closed safely")



import sys
import numpy as np
from collections import deque
from pynput import keyboard

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


class JointPlotter:
    def __init__(self, joint_names, history_len=500, render_stride=5):
        """
        joint_names: list[str]  (左右对称 joint 的“基名”，如 hip_pitch)
        history_len: 曲线窗口长度
        render_stride: 每多少 step 才 render 一次
        """
        self.joint_names = joint_names
        self.n = len(joint_names)
        self.history_len = history_len
        self.render_stride = render_stride

        self.step = 0
        self.enable_render = False

        # ===== flags（键盘线程 → 主线程）=====
        self.request_close = False
        self.request_force_render = False
        self._closed = False

        # ===== Qt Application（只建一次）=====
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication(sys.argv)

        # ===== 主窗口 =====
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("Joint Plotter")
        self.layout = QtWidgets.QGridLayout(self.win)

        # ===== 数据缓存（deque，O(1)）=====
        self.data = {
            "torque": [ {"L": deque(maxlen=history_len),
                          "R": deque(maxlen=history_len)} for _ in range(self.n) ],
            "pos":    [ {"L": deque(maxlen=history_len),
                          "R": deque(maxlen=history_len)} for _ in range(self.n) ],
            "vel":    [ {"L": deque(maxlen=history_len),
                          "R": deque(maxlen=history_len)} for _ in range(self.n) ],
        }

        # ===== 曲线句柄 =====
        self.lines = {
            "torque": [],
            "pos": [],
            "vel": [],
        }

        for i, name in enumerate(joint_names):
            for j, key in enumerate(["torque", "pos", "vel"]):
                plot = pg.PlotWidget(title=f"{name} | {key}")
                plot.showGrid(x=True, y=True)
                plot.addLegend(offset=(10, 10))

                line_l = plot.plot(pen=pg.mkPen("r", width=2), name="L")
                line_r = plot.plot(pen=pg.mkPen("b", width=2), name="R")

                self.lines[key].append((line_l, line_r))
                self.layout.addWidget(plot, i, j)

        self.win.resize(1600, 300 * self.n)
        self.win.show()

        # ===== Qt 定时器：只负责 GUI =====
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._qt_tick)
        self._timer.start(20)   # 50 Hz GUI，不抢焦点

        # ===== 键盘监听 =====
        self._start_keyboard_listener()

        print("[JointPlotter] ready | keys: p(toggle), l(refresh), q(quit)")

    # ------------------------------------------------------------------
    # pynput 键盘监听（子线程）
    # ------------------------------------------------------------------
    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == "p":
                    self.enable_render = not self.enable_render
                    print(f"[Plot] render = {self.enable_render}")

                elif key.char == "l":
                    print("[Plot] force refresh")
                    self.request_force_render = True

                elif key.char == "q":
                    print("[Plot] request close")
                    self.request_close = True
                    return False
            except AttributeError:
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    # ------------------------------------------------------------------
    # 外部接口（与你现在的代码完全一致）
    # ------------------------------------------------------------------
    def update(self, tau_l, tau_r, pos_l, pos_r, vel_l, vel_r):
        if self._closed:
            return

        self.step += 1

        for i in range(self.n):
            self.data["torque"][i]["L"].append(tau_l[i])
            self.data["torque"][i]["R"].append(tau_r[i])

            self.data["pos"][i]["L"].append(pos_l[i])
            self.data["pos"][i]["R"].append(pos_r[i])

            self.data["vel"][i]["L"].append(vel_l[i])
            self.data["vel"][i]["R"].append(vel_r[i])

        QtWidgets.QApplication.processEvents()
    # ------------------------------------------------------------------
    # Qt 主线程周期（安全）
    # ------------------------------------------------------------------
    def _qt_tick(self):
        if self._closed:
            return

        if self.request_close:
            self._do_close()
            return

        if self.request_force_render:
            self._render()
            self.request_force_render = False
            return

        if self.enable_render and self.step % self.render_stride == 0:
            self._render()

    # ------------------------------------------------------------------
    # 真正绘图（极快）
    # ------------------------------------------------------------------
    def _render(self):
        for i in range(self.n):
            for key in ["torque", "pos", "vel"]:
                yL = list(self.data[key][i]["L"])
                yR = list(self.data[key][i]["R"])
                x = np.arange(len(yL))

                line_l, line_r = self.lines[key][i]
                line_l.setData(x, yL)
                line_r.setData(x, yR)

    # ------------------------------------------------------------------
    # 安全关闭
    # ------------------------------------------------------------------
    def _do_close(self):
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self.win.close()
        print("[JointPlotter] closed safely")
