import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import os
from lab3_bestconfig_linear import bestconfig_fast_search

class BestConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BestConfig Fast Search GUI")
        self.file_path = ""

        # Budget input
        tk.Label(root, text="Budget:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.budget_entry = tk.Entry(root)
        self.budget_entry.insert(0, "100")
        self.budget_entry.grid(row=0, column=1, padx=5, pady=5)

        # File selection
        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.select_button = tk.Button(root, text="Select CSV File", command=self.select_file)
        self.select_button.grid(row=2, column=0, columnspan=2, pady=5)

        # Run button
        self.run_button = tk.Button(root, text="Run Optimization", command=self.run_optimization)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Result TreeView
        self.tree = ttk.Treeview(root, columns=("Performance",), show="headings")
        self.tree.heading("Performance", text="Performance")
        self.tree.grid(row=4, column=0, columnspan=2, sticky="nsew")

        # Best result
        self.best_label = tk.Label(root, text="Best Configuration:")
        self.best_label.grid(row=5, column=0, columnspan=2, pady=10)

        root.grid_rowconfigure(4, weight=1)
        root.grid_columnconfigure(1, weight=1)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path = path
            self.file_label.config(text=os.path.basename(path))

    def run_optimization(self):
        if not self.file_path:
            messagebox.showwarning("No File", "Please select a CSV file.")
            return

        try:
            budget = int(self.budget_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Budget must be an integer.")
            return

        output_file = "search_result_temp.csv"

        try:
            best_solution, best_perf = bestconfig_fast_search(
                self.file_path, budget, output_file
            )
            df = pd.read_csv(output_file)
            self.tree.delete(*self.tree.get_children())
            for _, row in df.iterrows():
                conf_str = ", ".join(map(str, row[:-1]))
                perf_str = f"{row[-1]:.4f}"
                self.tree.insert("", "end", values=(f"{conf_str} -> {perf_str}"))

            self.best_label.config(text=f"Best Configuration: {best_solution} | Performance: {best_perf:.4f}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BestConfigGUI(root)
    root.geometry("700x500")
    root.mainloop()
