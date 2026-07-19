REPORT = report/摩托车事故归因分析报告.md
BLOG = ~/src/blog/think
DEPS = --with pandas --with numpy --with seaborn --with matplotlib --with scikit-learn --with openpyxl --with pillow --with fonttools

all: gen-report sync-to-blog

# 生成报告
gen-report:
	uv run $(DEPS) python3 gen_report.py

# 同步到博客（Hugo 适配）
BLOG_ASSETS = ~/src/blog/static/images/motor-accident
BLOG_POST = ~/src/blog/content/think

sync-to-blog:
	mkdir -p $(BLOG_ASSETS) $(BLOG_POST)
	cp report/report_assets/*.png $(BLOG_ASSETS)/
	{ echo '+++'; \
	  echo 'title       = "摩托车事故归因分析报告"'; \
	  echo 'date        = '"$$(date '+%Y-%m-%dT%H:%M:%S+08:00')"; \
	  echo 'author      = "weaming"'; \
	  echo 'description = "基于孟加拉和 FARS 数据，分析摩托车事故的关键归因因素"'; \
	  echo '+++'; \
	  echo ''; \
	  tail -n +2 report/摩托车事故归因分析报告.md; } | \
	  sed 's|report_assets/|/images/motor-accident/|g' > $(BLOG_POST)/摩托车事故归因分析报告.md

