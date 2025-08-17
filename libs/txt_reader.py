from PyPDF2 import PdfReader
import os

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f" Plik TXT nie został znaleziony: {file_path}")
        return ""

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f" Błąd podczas odczytu PDF: {e}")
        return ""

def get_text_interactive():
    while True:
        print("=== Wybierz źródło tekstu ===")
        print("1 - Wczytaj z pliku TXT")
        print("2 - Wczytaj z pliku PDF")
        print("3 - Wpisz tekst ręcznie")
        choice = input("Twój wybór (1/2/3): ").strip()

        if choice == "1":
            path = input(" Podaj ścieżkę do pliku TXT: ").strip()
            if not path.lower().endswith(".txt"):
                path += ".txt"
            text = extract_text_from_txt(path)
            if text:
                return text

        elif choice == "2":
            path = input(" Podaj ścieżkę do pliku PDF: ").strip()
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            text = extract_text_from_pdf(path)
            if text:
                return text

        elif choice == "3":
            print("️ Wpisz swój tekst (zakończ ENTER):")
            return input("> ")

        else:
            print(" Nieprawidłowy wybór. Spróbuj ponownie.\n")
