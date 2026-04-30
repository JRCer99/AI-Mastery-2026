import json
from datetime import datetime

def load_portfolio_data(filename="portfolio_data.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ {filename} not found! Please create it first.")


def generate_markdown(data):
    md = f"""# {data['name']}

![Profile Views](https://komarev.com/ghpvc/?username={data['github']}&color=brightgreen)

## 👋 About Me
{data['bio']}

- 📍 **Location:** {data['location']}
- ✉️ **Email:** {data['email']}
- 🔗 **GitHub:** [{data['github']}](https://github.com/{data['github']})
- 💼 **LinkedIn:** {f"[{data['linkedin']}](https://linkedin.com/in/{data['linkedin']})" if data.get('linkedin') else 'N/A'}

## 🛠️ Skills
"""

    for skill in data['skills']:
        md += f"- {skill}\n"

    md += "\n## 🎓 Education\n"
    for edu in data['education']:
        md += f"- **{edu['degree']} in {edu['field']}** — {edu['school']} ({edu['year']})\n"

    md += "\n## 🚀 Projects\n"
    for project in data['projects']:
        md += f"### {project['name']}\n"
        md += f"{project['description']}\n"
        md += f"**Tech:** {', '.join(project['tech'])}\n"
        md += f"[View Project]({project['link']})\n\n"

    md += f"\n---\n*Last updated: {datetime.now().strftime('%B %d, %Y')}*\n"
    return md


def generate_html(data):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['name']} - Portfolio</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; }}
        h1, h2 {{ color: #2c3e50; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{data['name']}</h1>
        <h2>{data['title']}</h2>
        <p>{data['bio']}</p>
        <p>📍 {data['location']} | ✉️ {data['email']}</p>
        
        <h2>🛠️ Skills</h2>
        <ul>
"""
    for skill in data['skills']:
        html += f"            <li>{skill}</li>\n"
    html += """        </ul>

        <h2>🎓 Education</h2>
        <ul>
"""
    for edu in data['education']:
        html += f"            <li><strong>{edu['degree']} in {edu['field']}</strong> — {edu['school']} ({edu['year']})</li>\n"
    html += """        </ul>

        <h2>🚀 Projects</h2>
"""
    for project in data['projects']:
        html += f"""        <h3>{project['name']}</h3>
        <p>{project['description']}</p>
        <p><strong>Tech:</strong> {', '.join(project['tech'])}</p>
        <a href="{project['link']}">View Project →</a><br><br>
"""
    html += f"""        <hr>
        <p><small>Last updated: {datetime.now().strftime('%B %d, %Y')}</small></p>
    </div>
</body>
</html>"""
    return html


def main():
    data = load_portfolio_data()
    
    # Generate Markdown
    markdown_content = generate_markdown(data)
    with open("portfolio_README.md", "w") as f:
        f.write(markdown_content)
    print("✅ Generated portfolio_README.md")

    # Generate HTML
    html_content = generate_html(data)
    with open("index.html", "w") as f:
        f.write(html_content)
    print("✅ Generated index.html")

    print("\n🎉 Portfolio files generated! Open index.html in your browser.")


if __name__ == "__main__":
    main()