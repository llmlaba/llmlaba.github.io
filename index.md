---
layout: home
title: "Main page"
---

# Articles
Occasionally, I write about LLM AI technologies that interest me on this blog.  

## GPU
{% assign hw_articles = site.categories.gpu | default: empty %}
{% for article in hw_articles %}
- [{{ article.title }}]({{ article.url | relative_url }}) — {{ article.date | date: "%d.%m.%Y" }}
{% endfor %}

## Software
{% assign sw_articles = site.categories.software | default: empty %}
{% for article in sw_articles %}
- [{{ article.title }}]({{ article.url | relative_url }}) — {{ article.date | date: "%d.%m.%Y" }}
{% endfor %}
