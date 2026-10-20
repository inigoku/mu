import os
from fpdf import FPDF
import markdown

def md_to_pdf(input_file, output_file):
    # Leer el contenido del archivo markdown
    with open(input_file, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Como fpdf2 no renderiza Markdown directamente,
    # haremos una conversión simple de texto plano para evitar errores
    # de formato complejos que requerirían un motor HTML completo.

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Usar una fuente estándar que soporte la mayoría de los caracteres
    # Nota: fpdf2 usa fuentes estándar por defecto (Helvetica, Times, Courier)
    pdf.set_font("Helvetica", size=12)

    # Dividir el texto en líneas para procesarlo
    lines = md_text.split('\n')

    for line in lines:
        # Manejo muy básico de Markdown
        if line.startswith('# '):
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0, 10, line[2:].strip(), ln=True)
            pdf.set_font("Helvetica", size=12)
        elif line.startswith('## '):
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, line[3:].strip(), ln=True)
            pdf.set_font("Helvetica", size=12)
        elif line.startswith('### '):
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(0, 10, line[4:].strip(), ln=True)
            pdf.set_font("Helvetica", size=12)
        else:
            # multi_cell es necesario para que el texto haga salto de línea automático
            pdf.multi_cell(0, 10, line)

        pdf.ln(2)

    pdf.output(output_file)
    print(f"Archivo convertido exitosamente a: {output_file}")

if __name__ == "__main__":
    input_md = "El Horizonte Interior Ensayo completo caps53-54.md"
    output_pdf = "El Horizonte Interior Ensayo completo caps53-54.pdf"
    md_to_pdf(input_md, output_pdf)
