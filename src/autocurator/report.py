from pathlib import Path

from jinja2 import Template

# Repo layout: <root>/src/autocurator/report.py  ->  <root>/templates/report.html
_DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "report.html"


def render_report(out_path: str, context: dict, template_path: str | None = None):
    template_file = Path(template_path) if template_path else _DEFAULT_TEMPLATE
    with open(template_file, encoding="utf-8") as f:
        tpl = Template(f.read())
    html = tpl.render(**context)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
