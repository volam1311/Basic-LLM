# Course website

The site is built with [MkDocs](https://www.mkdocs.org/) and the Material theme from the notes in `Module */` and the top-level `README.md` (which becomes the home page). The notes are not moved: `prepare.sh` copies them into `site/docs/` before each build.

Preview locally:

```bash
pip install -r requirements-site.txt
./site/prepare.sh
mkdocs serve --config-file site/mkdocs.yml
```

Re-run `prepare.sh` after editing a note. When you add a lesson, also add it to `nav` in `site/mkdocs.yml`.

`.github/workflows/deploy-site.yml` builds the site on every push to `main` and deploys it to GitHub Pages. Pull requests are only built, so a broken link fails the check (the build runs in strict mode).
