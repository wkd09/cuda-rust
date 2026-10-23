# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

project = "cuda-oxide 한국어 번역"
author = "CUDA Rust Korean translation contributors"
master_doc = "index"
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = ["myst_parser", "sphinx_copybutton", "sphinx_design", "sphinx_sitemap"]
myst_enable_extensions = ["colon_fence", "deflist", "tasklist", "attrs_inline"]

html_theme = "pydata_sphinx_theme"
html_title = "cuda-oxide 한국어 번역"
html_baseurl = "https://wkd09.github.io/cuda-rust/cuda-oxide/ko/"
html_show_sourcelink = False
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wkd09/cuda-rust",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}
