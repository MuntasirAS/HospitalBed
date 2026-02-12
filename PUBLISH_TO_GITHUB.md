
# Publish to GitHub — Quick Guide

1) Create a new repo on GitHub (empty).
2) In your terminal:
```bash
git init
git add .
git commit -m "feat: initial modular refactor of NHS MAS"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Optionally enable Actions (CI) with lint/test workflows.
