"""Capturas, informe local y láminas de respaldo de Nery.

Instalar herramientas de exportación:
  python -m pip install playwright==1.55.0 markdown==3.9
Arrancar la app en localhost:8502 y ejecutar este archivo.
Usa Edge instalado en Windows; no modifica Google Drive.
"""
from pathlib import Path
import re
import markdown
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
SECCIONES = RAIZ / "informe/resultados"
FIGURAS = RAIZ / "results/figures"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        for ruta, imagen, texto in [
            ("", "04_app_inicio.png", "¿Qué clientes vuelven a comprar?"),
            ("Explorar_datos", "04_app_explorar.png", "Explorar los datos limpios"),
            ("Rendimiento", "04_app_rendimiento.png", "Mayor AUC temporal:"),
        ]:
            page.goto("http://127.0.0.1:8502/" + ruta)
            page.get_by_text(texto, exact=False).first.wait_for(timeout=60000)
            if ruta:
                page.locator(".js-plotly-plot").first.wait_for(timeout=60000)
                page.wait_for_function("document.querySelectorAll('.js-plotly-plot .main-svg').length > 0")
            page.screenshot(path=str(FIGURAS / imagen))
            if ruta == "Rendimiento":
                page.get_by_text("Errores al cambiar el umbral", exact=True).scroll_into_view_if_needed()
                page.screenshot(path=str(FIGURAS / "04_app_umbral.png"))
                page.get_by_text("Mostrar información de rendimiento", exact=True).click()
                page.get_by_text("La información de rendimiento está oculta.", exact=True).wait_for()
                assert page.locator(".js-plotly-plot").count() == 0
                page.screenshot(path=str(FIGURAS / "04_app_oculto.png"))
        # Cada sección se conserva como aporte original. Se compila sin sobrescribirla.
        faltan = [x for x in ["01_introduccion_y_objetivos.md", "04_datos_y_limpieza.md"] if not (SECCIONES / x).exists()]
        aviso = "# Proyecto 2 · compilación local para revisión\n\n"
        aviso += "**Estado:** pendiente de integración en Google Drive, revisión cruzada de citas APA y prueba externa de uso.\n\n"
        aviso += "**Secciones faltantes del equipo:** " + ", ".join(faltan) + ". No se reemplazaron con contenido atribuido a otros integrantes.\n\n"
        bloques = [markdown.markdown(aviso)]
        for archivo in sorted(SECCIONES.glob("*.md")):
            if archivo.name.startswith("verificacion"):
                continue
            texto = archivo.read_text(encoding="utf-8")
            # Reubicar imágenes relativas respecto al archivo fuente.
            texto = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', lambda m: f'![{m[1]}](../{(archivo.parent / m[2]).resolve().relative_to(RAIZ).as_posix()})', texto)
            bloques.append('<section>' + markdown.markdown(texto, extensions=["tables", "fenced_code"]) + '</section>')
        css = """@page {size:A4; margin:25.4mm} body{font-family:'Times New Roman',serif;font-size:12pt;line-height:1.4;color:#222} p{text-align:justify} h1,h2,h3{break-after:avoid} section{break-before:page} img{max-width:100%;max-height:210mm;object-fit:contain} table{border-collapse:collapse;width:100%;font-size:9pt} td,th{border:1px solid #aaa;padding:4px} pre,code{white-space:pre-wrap;overflow-wrap:anywhere;font-size:9pt} a{color:#0072B2} """
        html = '<!doctype html><html lang="es"><meta charset="utf-8"><style>' + css + '</style><body>' + '\n'.join(bloques) + '</body></html>'
        destino = RAIZ / "informe/resultados_revision_local.html"
        destino.write_text(html, encoding="utf-8")
        page.goto(destino.as_uri())
        page.evaluate("Promise.all(Array.from(document.images).map(i => i.decode().catch(() => {})))")
        page.pdf(path=str(RAIZ / "informe/resultados_revision_local.pdf"), print_background=True, prefer_css_page_size=True)
        slides = [("El historial del producto se asocia con mayor recompra", "04_app_explorar.png"),
                  ("La regresión logística alcanza AUC 0.647 en la semana de prueba", "04_app_rendimiento.png"),
                  ("El umbral cambia los errores, sin cambiar el AUC", "04_app_umbral.png"),
                  ("El usuario decide si quiere ver el rendimiento", "04_app_oculto.png")]
        contenido = ''.join(f'<section><h1>{titulo}</h1><img src="../results/figures/{imagen}"></section>' for titulo, imagen in slides)
        laminas = RAIZ / "informe/laminas_nery.html"
        laminas.write_text('<!doctype html><html lang="es"><meta charset="utf-8"><style>@page{size:landscape;margin:10mm} body{font-family:Arial;color:#222} section{break-after:page} h1{font-size:22pt;color:#0072B2} img{max-width:100%;max-height:160mm}</style><body>'+contenido+'</body></html>', encoding="utf-8")
        page.goto(laminas.as_uri())
        page.evaluate("Promise.all(Array.from(document.images).map(i => i.decode().catch(() => {})))")
        page.pdf(path=str(RAIZ / "informe/laminas_nery.pdf"), print_background=True, prefer_css_page_size=True)
        browser.close()
    print("Capturas, compilación local y láminas PDF generadas. La revisión local no es el informe final de Drive.")

if __name__ == "__main__":
    main()
