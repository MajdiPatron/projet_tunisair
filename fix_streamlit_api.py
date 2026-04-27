"""Script utilitaire: corrige les API Streamlit depreciees dans les fichiers app/"""
import os

app_dir = os.path.join(os.path.dirname(__file__), "app")
for fname in os.listdir(app_dir):
    if fname.endswith(".py"):
        path = os.path.join(app_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new = content.replace("use_container_width=True", "width='stretch'")
        new = new.replace("use_container_width=False", "width='content'")
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        if new != content:
            print(f"Fixed: {fname}")
print("Done.")
