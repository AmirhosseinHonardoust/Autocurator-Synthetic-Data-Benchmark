from jinja2 import Template

def render_report(template_path: str, out_path: str, context: dict):
    with open(template_path, "r", encoding="utf-8") as f:
        tpl = Template(f.read())
    html = tpl.render(**context)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
