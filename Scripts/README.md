# Hand Tracking Setup Script

## 📋 Vereisten

- Python **3.10** (3.12 en 3.13 worden niet ondersteund)
- pip
- VS Code of een andere Python IDE

> **Belangrijk:** Alle `.task` bestanden moeten in dezelfde map staan als `hand2.py`.

---

## ⚡ Installatie

Open een terminal in je projectmap en voer de volgende commando's uit:

### 1. Virtuele omgeving aanmaken en activeren
```bash
py -3.10 -m venv venv
venv/Scripts/Activate
```

### 2. Problemen met toestemming oplossen (indien nodig)

Als je een foutmelding krijgt over execution policy, voer dan uit:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Libraries installeren
```bash
pip install opencv-python mediapipe pyautogui numpy
```

---

## ✅ Klaar!

Je bent nu klaar om te beginnen met hand tracking.
