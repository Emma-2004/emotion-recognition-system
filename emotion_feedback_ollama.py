import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import cv2
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from deepface import DeepFace
import requests
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

class EmotionFeedbackSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("学习状态情绪反馈系统V1.0")
        self.root.geometry("1400x850")
        self.root.minsize(1200, 750)

        self.camera_running = False
        self.analyzing = False
        self.cap = None
        self.current_frame = None
        self.face_detector = MTCNN()
        self.history = []
        self.start_time = None
        self.break_reminder_enabled = True
        self.break_interval = 30

        self.color_primary = "#1677ff"
        self.color_bg = "#f5f7fa"
        self.color_card = "#ffffff"
        self.color_text = "#333333"
        self.color_text_secondary = "#666666"

        self.create_layout()
        self.check_break_reminder()

    def create_layout(self):
        self.sidebar = tk.Frame(self.root, bg=self.color_primary, width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="学习状态情绪反馈系统", font=("Microsoft YaHei", 14, "bold"),
                 bg=self.color_primary, fg="white").pack(fill="x", padx=15, pady=20)

        # 【关键修改】移除侧边栏"数据分析"按钮，其余全部保留
        menu_items = [
            ("实时监控", self.show_monitor_page),
            ("学习记录", self.show_record_page),
            ("系统设置", self.show_setting_page),
            ("导出数据", self.export_history),
            ("停止监控", self.stop)
        ]
        for text, cmd in menu_items:
            btn = tk.Button(self.sidebar, text=text, font=("Microsoft YaHei", 10),
                            bg=self.color_primary, fg="white", bd=0, relief="flat",
                            anchor="w", padx=20, pady=12, command=cmd, cursor="hand2")
            btn.pack(fill="x")

        self.main_container = tk.Frame(self.root, bg=self.color_bg)
        self.main_container.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        self.top_bar = tk.Frame(self.main_container, bg=self.color_primary, height=60)
        self.top_bar.pack(fill="x", pady=(0, 15))
        self.top_bar.pack_propagate(False)

        tk.Label(self.top_bar, text="学习状态情绪反馈系统V1.0", font=("Microsoft YaHei", 14, "bold"),
                 bg=self.color_primary, fg="white").pack(side="left", padx=20)

        self.time_label = tk.Label(self.top_bar, text="学习时长：00:00:00",
                                    font=("Microsoft YaHei", 11), bg=self.color_primary, fg="white")
        self.time_label.pack(side="right", padx=20)

        self.content_area = tk.Frame(self.main_container, bg=self.color_bg)
        self.content_area.pack(fill="both", expand=True)

        self.monitor_page = tk.Frame(self.content_area, bg=self.color_bg)
        self.monitor_page.pack(fill="both", expand=True)

        self.left_col = tk.Frame(self.monitor_page, bg=self.color_card, padx=15, pady=15)
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(self.left_col, text="实时摄像头监控", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w", pady=(0, 10))
        self.camera_label = tk.Label(self.left_col, text="摄像头未开启",
                                     font=("Microsoft YaHei", 12), bg=self.color_card)
        self.camera_label.pack(fill="both", expand=True, pady=10)

        control_frame = tk.Frame(self.left_col, bg=self.color_card)
        control_frame.pack(fill="x", pady=15)
        btn_style = {"font": ("Microsoft YaHei", 10), "bg": "#1677ff", "fg": "white",
                      "padx": 12, "pady": 6, "bd": 0, "relief": "flat"}
        # 监控页情绪分析按钮保留，功能完整可用
        tk.Button(control_frame, text="开始监控", command=self.start, **btn_style).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(control_frame, text="情绪分析", command=self.analyze, **btn_style).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(control_frame, text="重置数据", command=self.reset, **btn_style).pack(side="left", padx=5, fill="x", expand=True)

        self.right_col = tk.Frame(self.monitor_page, bg=self.color_card, padx=15, pady=15)
        self.right_col.pack(side="right", fill="both", expand=True)

        tk.Label(self.right_col, text="人脸检测区域", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w", pady=(0, 10))
        self.face_label = tk.Label(self.right_col, text="等待分析...", font=("Microsoft YaHei", 10), bg=self.color_card)
        self.face_label.pack(pady=10)

        tk.Label(self.right_col, text="学习状态概览", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w", pady=(0, 10))
        self.status_label = tk.Label(self.right_col, text="当前状态：等待监控启动", font=("Microsoft YaHei", 11), bg=self.color_card)
        self.status_label.pack(anchor="w", pady=5)

        tk.Label(self.right_col, text="情绪状态分布", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w", pady=(0, 10))
        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.fig.patch.set_facecolor(self.color_card)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_col)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(0, 10))

        tk.Label(self.right_col, text="AI学习建议", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w", pady=(10, 5))
        self.feedback_text = scrolledtext.ScrolledText(self.right_col, height=8, font=("Microsoft YaHei", 10),
                                                        bg="#f9f9f9", relief="flat", bd=1, wrap=tk.WORD)
        self.feedback_text.pack(fill="both", expand=True, pady=(0, 5))
        self.feedback_text.insert(tk.END, "等待分析后生成学习建议...")

        self.record_page = tk.Frame(self.content_area, bg=self.color_bg)
        record_card = tk.Frame(self.record_page, bg=self.color_card, padx=20, pady=20)
        record_card.pack(fill="both", expand=True)
        tk.Label(record_card, text="学习记录历史", font=("Microsoft YaHei", 14, "bold"),
                 bg=self.color_card).pack(anchor="w", pady=(0, 15))

        record_frame = tk.Frame(record_card, bg=self.color_card)
        record_frame.pack(fill="both", expand=True)
        columns = ("时间", "情绪", "专注度", "分心", "疲倦", "焦虑", "放松")
        self.record_tree = ttk.Treeview(record_frame, columns=columns, show="headings", height=15)
        for col in columns:
            self.record_tree.heading(col, text=col)
            self.record_tree.column(col, width=100, anchor="center")
        self.record_tree.pack(fill="both", expand=True)

        record_btn_frame = tk.Frame(record_card, bg=self.color_card)
        record_btn_frame.pack(fill="x", pady=15)
        tk.Button(record_btn_frame, text="刷新记录", command=self.refresh_record, **btn_style).pack(side="left", padx=5)
        tk.Button(record_btn_frame, text="清空记录", command=self.clear_record,
                  font=("Microsoft YaHei", 10), bg="#f5222d", fg="white", padx=12, pady=6, bd=0, relief="flat").pack(side="left", padx=5)
        tk.Button(record_btn_frame, text="导出CSV", command=self.export_history,
                  font=("Microsoft YaHei", 10), bg="#52c41a", fg="white", padx=12, pady=6, bd=0, relief="flat").pack(side="right", padx=5)

        self.setting_page = tk.Frame(self.content_area, bg=self.color_bg)
        setting_card = tk.Frame(self.setting_page, bg=self.color_card, padx=20, pady=20)
        setting_card.pack(fill="both", expand=True)
        tk.Label(setting_card, text="系统设置", font=("Microsoft YaHei", 14, "bold"),
                 bg=self.color_card).pack(anchor="w", pady=(0, 20))

        break_frame = tk.Frame(setting_card, bg=self.color_card)
        break_frame.pack(fill="x", pady=10)
        tk.Label(break_frame, text="学习休息提醒", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w")
        self.break_var = tk.BooleanVar(value=self.break_reminder_enabled)
        tk.Checkbutton(break_frame, text="开启休息提醒", variable=self.break_var,
                       command=self.toggle_break, font=("Microsoft YaHei", 10), bg=self.color_card).pack(anchor="w", pady=5)

        interval_frame = tk.Frame(break_frame, bg=self.color_card)
        interval_frame.pack(anchor="w", pady=5)
        tk.Label(interval_frame, text="提醒间隔（分钟）：", font=("Microsoft YaHei", 10),
                 bg=self.color_card).pack(side="left")
        self.interval_entry = tk.Entry(interval_frame, font=("Microsoft YaHei", 10), width=10)
        self.interval_entry.insert(0, str(self.break_interval))
        self.interval_entry.pack(side="left", padx=5)
        tk.Button(interval_frame, text="保存", command=self.save_interval, **btn_style).pack(side="left")

        ui_frame = tk.Frame(setting_card, bg=self.color_card)
        ui_frame.pack(fill="x", pady=15)
        tk.Label(ui_frame, text="界面设置", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w")
        tk.Button(ui_frame, text="切换深色模式", command=self.toggle_dark_mode, **btn_style).pack(anchor="w", pady=5)

        about_frame = tk.Frame(setting_card, bg=self.color_card)
        about_frame.pack(fill="x", pady=15)
        tk.Label(about_frame, text="关于系统", font=("Microsoft YaHei", 12, "bold"),
                 bg=self.color_card).pack(anchor="w")
        tk.Label(about_frame, text="学习状态情绪反馈系统 V1.0", font=("Microsoft YaHei", 10),
                 bg=self.color_card).pack(anchor="w", pady=5)

        self.show_monitor_page()

    def show_monitor_page(self):
        self.record_page.pack_forget()
        self.setting_page.pack_forget()
        self.monitor_page.pack(fill="both", expand=True)

    def show_record_page(self):
        self.monitor_page.pack_forget()
        self.setting_page.pack_forget()
        self.refresh_record()
        self.record_page.pack(fill="both", expand=True)

    def show_setting_page(self):
        self.monitor_page.pack_forget()
        self.record_page.pack_forget()
        self.setting_page.pack(fill="both", expand=True)

    def refresh_record(self):
        for item in self.record_tree.get_children():
            self.record_tree.delete(item)
        for record in self.history:
            self.record_tree.insert("", tk.END, values=(
                record["time"], record["emotion"], f"{record['focus']}%",
                f"{record['distracted']}%", f"{record['tired']}%",
                f"{record['anxious']}%", f"{record['relaxed']}%"
            ))

    def clear_record(self):
        if messagebox.askyesno("确认", "确定清空所有学习记录？"):
            self.history.clear()
            self.refresh_record()

    def toggle_break(self):
        self.break_reminder_enabled = self.break_var.get()

    def save_interval(self):
        try:
            interval = int(self.interval_entry.get())
            if interval < 5:
                messagebox.showwarning("错误", "间隔不小于5分钟")
                return
            self.break_interval = interval
            messagebox.showinfo("成功", "设置完成")
        except:
            messagebox.showwarning("错误", "输入有效数字")

    def toggle_dark_mode(self):
        if self.color_bg == "#f5f7fa":
            self.color_bg = "#1e1e1e"
            self.color_card = "#2d2d2d"
            self.color_text = "#ffffff"
            self.color_text_secondary = "#cccccc"
        else:
            self.color_bg = "#f5f7fa"
            self.color_card = "#ffffff"
            self.color_text = "#333333"
            self.color_text_secondary = "#666666"
        self.root.config(bg=self.color_bg)

    def check_break_reminder(self):
        def run():
            while True:
                if self.camera_running and self.break_reminder_enabled:
                    elapsed = time.time() - self.start_time
                    if elapsed >= self.break_interval * 60:
                        self.root.after(0, lambda: messagebox.showinfo("休息提醒", "建议休息5分钟"))
                        self.start_time = time.time()
                time.sleep(60)
        threading.Thread(target=run, daemon=True).start()

    def start(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            self.camera_running = True
            self.start_time = time.time()
            self.status_label.config(text="当前状态：监控中")
            self.update_camera()
            self.start_timer()

    def stop(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_label.config(image="", text="已停止")
        self.status_label.config(text="已停止")

    def reset(self):
        self.stop()
        self.time_label.config(text="学习时长：00:00:00")
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.insert(tk.END, "等待分析")
        self.face_label.config(image="", text="等待分析")
        self.ax.clear()
        self.canvas.draw()

    def update_camera(self):
        if self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((500, 350))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk, text="")
            self.root.after(30, self.update_camera)

    def analyze(self):
        if not self.camera_running or self.current_frame is None:
            messagebox.showwarning("提示", "先开启摄像头")
            return
        threading.Thread(target=self.run_analyze, daemon=True).start()

    def run_analyze(self):
        try:
            frame = self.current_frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector.detect_faces(rgb)
            if not faces:
                self.status_label.config(text="未检测到人脸")
                return
            x, y, w, h = faces[0]['box']
            face_img = rgb[y:y+h, x:x+w]
            self.show_face(face_img)
            emo = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)[0]
            emotion = emo['dominant_emotion']
            scores = emo['emotion']
            focus = scores.get('neutral', 0)*0.85 + scores.get('happy', 0)*0.1
            tired = scores.get('sad', 0)*0.6 + scores.get('fear', 0)*0.4
            distracted = scores.get('surprise', 0)*0.7
            anxious = scores.get('angry', 0)*0.9
            relaxed = scores.get('happy', 0)*0.95
            total = focus + tired + distracted + anxious + relaxed
            if total > 0:
                focus = focus/total*100
                tired = tired/total*100
                distracted = distracted/total*100
                anxious = anxious/total*100
                relaxed = relaxed/total*100
            self.show_chart(['专注','分心','疲倦','焦虑','放松'], [focus,distracted,tired,anxious,relaxed])
            record = {
                "time": datetime.now().strftime("%H:%M:%S"), "emotion": emotion,
                "focus": round(focus,1), "distracted": round(distracted,1),
                "tired": round(tired,1), "anxious": round(anxious,1), "relaxed": round(relaxed,1)
            }
            self.history.append(record)
            self.status_label.config(text=f"{emotion} | 专注{focus:.1f}%")
            self.get_feedback(record)
        except Exception as e:
            self.status_label.config(text=f"分析失败：{str(e)}")

    def show_face(self, img):
        img = Image.fromarray(img).resize((180,180))
        imgtk = ImageTk.PhotoImage(image=img)
        self.face_label.imgtk = imgtk
        self.face_label.config(image=imgtk, text="")

    def show_chart(self, labels, values):
        self.ax.clear()
        self.ax.bar(labels, values, color=['#1677ff','#faad14','#f5222d','#722ed1','#52c41a'])
        self.ax.set_ylim(0,100)
        self.canvas.draw()

    def get_feedback(self, data):
        focus = data["focus"]
        distracted = data["distracted"]
        tired = data["tired"]
        anxious = data["anxious"]
        relaxed = data["relaxed"]
        emotion = data["emotion"]

        if tired > 40:
            prompt = f"用户学习疲倦度高达{tired}%，专注仅{focus}%，请给出简短休息调整学习建议，100字以内"
        elif distracted > 35:
            prompt = f"用户分心程度{distracted}%，专注力不足，给出提升专注、减少走神的学习小技巧，100字以内"
        elif anxious > 30:
            prompt = f"用户焦虑情绪{anxious}%，心态浮躁，给出舒缓焦虑、平稳学习心态的建议，100字以内"
        elif relaxed > 70 and focus > 25:
            prompt = f"用户当前放松状态充足、情绪稳定，给出维持高效学习节奏的建议，100字以内"
        else:
            prompt = f"当前情绪{emotion}，专注{focus}%，综合状态平稳，给出优化学习效率的简短建议，100字以内"

        try:
            res = requests.post("http://localhost:11434/api/generate",
                                json={"model":"phi3","prompt":prompt}, timeout=10)
            text = res.json()["response"]
        except Exception as e:
            if tired > 40:
                text = "疲倦指标偏高，建议起身远眺5分钟，拉伸放松眼部与肩颈，短暂休息后再继续学习。"
            elif distracted > 35:
                text = "容易分心，可采用25分钟学习+5分钟休息的番茄工作法，隔绝手机等干扰源。"
            elif anxious > 30:
                text = "焦虑情绪明显，深呼吸3分钟梳理学习任务，拆分小目标降低心理压力。"
            elif relaxed > 70:
                text = "当前心态放松稳定，保持现有学习节奏，可适度增加难点内容提升学习深度。"
            else:
                text = "学习状态平稳，定期复盘知识点，合理分配各科学习时长，稳步提升专注度。"
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.insert(tk.END, text)

    def start_timer(self):
        def run():
            while self.camera_running:
                elapsed = time.time() - self.start_time
                h = int(elapsed//3600)
                m = int((elapsed%3600)//60)
                s = int(elapsed%60)
                self.time_label.config(text=f"学习时长：{h:02d}:{m:02d}:{s:02d}")
                time.sleep(1)
        self.timer_thread = threading.Thread(target=run, daemon=True)
        self.timer_thread.start()

    def export_history(self):
        if not self.history:
            messagebox.showinfo("提示", "暂无记录")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV文件","*.csv")])
        if path:
            with open(path,"w",newline="",encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=self.history[0].keys())
                w.writeheader()
                w.writerows(self.history)
            messagebox.showinfo("成功", "导出完成")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionFeedbackSystem(root)
    root.mainloop()