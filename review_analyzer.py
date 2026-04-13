import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Конфигурация
MAXLEN = 150
LABELS = {0: " Positive (Положительный)", 1: " Negative (Отрицательный)", 2: " Neutral (Нейтральный)"}
COLORS = {0: "green", 1: "red", 2: "#d97706"}

# АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ПУТЕЙ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_lstm_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
EMBED_PATH = os.path.join(BASE_DIR, "embed_matrix.npy")


def load_artifacts():  # ← ВАЖНО: не load_model!
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Токенизатор не найден: {TOKENIZER_PATH}")
        if not os.path.exists(EMBED_PATH):
            raise FileNotFoundError(f"Эмбеддинги не найдены: {EMBED_PATH}")

        model = load_model(MODEL_PATH)  # ← Теперь это TensorFlow Keras
        with open(TOKENIZER_PATH, "rb") as file:
            tokenizer = pickle.load(file)
        embed_matrix = np.load(EMBED_PATH)
        return model, tokenizer, embed_matrix
    except Exception as e:
        messagebox.showerror("Ошибка загрузки", f"Ошибка: {e}")
        return None, None, None


# Загрузка артефактов
model, tokenizer, embed_matrix = load_artifacts()


def show_context_menu(event):
    # Создает и показывает меню при нажатии правой кнопки мыши
    menu = tk.Menu(text_input, tearoff=0)
    # Вызываем стандартные системные события копирования и вставки
    menu.add_command(label="Копировать", command=lambda: text_input.event_generate('<<Copy>>'))
    menu.add_command(label="Вставить", command=lambda: text_input.event_generate('<<Paste>>'))
    menu.post(event.x_root, event.y_root)

# Логика предсказания
def predict_review():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Внимание", "Введите текст отзыва!")
        return

    # Блокируем кнопку на время предсказания
    btn.config(state=tk.DISABLED)
    root.update()

    try:
        # 1. Токенизация (точно как при обучении)
        seq = tokenizer.texts_to_sequences([text])
        if not seq or not seq[0]:
            result_label.config(text=" Текст не содержит слов из словаря", fg="gray")
            return

        # 2. Выравнивание длины (padding/truncating)
        padded = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')

        # 3. Проекция в 8D-эмбеддинги
        X = embed_matrix[padded[0]]  # форма (60, 8)
        X = np.expand_dims(X, axis=0)  # форма (1, 60, 8) для батча

        # 4. Инференс
        probs = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx] * 100)

        # 5. Вывод
        result_label.config(text=f"{LABELS[pred_idx]}\nУверенность: {confidence:.1f}%", fg=COLORS[pred_idx])

    except Exception as e:
        messagebox.showerror("Ошибка предсказания", f"Произошла ошибка: {e}")
    finally:
        btn.config(state=tk.NORMAL)

# Интерфейс
if model:
    root = tk.Tk()
    root.title("Анализатор тональности отзывов (RNN)")
    root.geometry("580x420")
    root.resizable(False, False)
    root.configure(bg="#f8f9fa")

    # Заголовок
    ttk.Label(root, text="Введите текст отзыва для анализа:", font=("Segoe UI", 12, "bold"), background="#f8f9fa").pack(pady=(20, 5))

    # Поле ввода
    text_input = tk.Text(root, height=7, width=MAXLEN, font=("Segoe UI", 11), bg="#ffffff", bd=1, relief="solid")
    text_input.pack(pady=5, padx=25)
    text_input.focus_set()
    text_input.bind("<Button-3>", show_context_menu)

    # Кнопка
    btn = ttk.Button(root, text=" Анализировать тональность", command=predict_review)
    btn.pack(pady=12)

    # ИСПРАВЛЕНИЕ: Используем tk.Label вместо ttk.Label для поддержки fg
    result_label = tk.Label(root, text="Здесь появится результат...", font=("Segoe UI", 13, "bold"),
                            background="#f8f9fa", justify="center")
    result_label.pack(pady=10)

    # Статус-бар
    status = ttk.Label(root, text=f" Модель загружена | Точность: 57.4% | Maxlen: {MAXLEN}",
                       relief=tk.SUNKEN, anchor="w", font=("Segoe UI", 9))
    status.pack(side=tk.BOTTOM, fill=tk.X)

    root.bind("<Control-Return>", lambda e: predict_review())
    root.mainloop()