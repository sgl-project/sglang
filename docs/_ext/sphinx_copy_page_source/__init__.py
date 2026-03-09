"""
Sphinx extension to add a "Copy" button that copies the current page's
Markdown source to clipboard. Works alongside sphinx_book_theme's download
button (.md open in new tab, PDF).
"""

import os

__version__ = "0.1.0"


def setup(app):
    ext_dir = os.path.dirname(__file__)
    app.config.html_static_path.append(ext_dir)
    app.add_js_file("copy_page_source.js")
    app.add_css_file("copy_page_source.css")
    return {"version": __version__, "parallel_read_safe": True}
