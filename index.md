---
layout: home
title: "Main page"
---

# Articles
Occasionally, I write about LLM AI technologies that interest me on this blog.  

## GPU
{% for article in site.articles %}
  {% if article.categories contains "gpu" %}
- [{{ article.title }}]({{ article.url | relative_url }}) — {{ article.date | date: "%d.%m.%Y" }}
  {% endif %}
{% endfor %}

## Software
{% for article in site.articles %}
  {% if article.categories contains "software" %}
- [{{ article.title }}]({{ article.url | relative_url }}) — {{ article.date | date: "%d.%m.%Y" }}
  {% endif %}
{% endfor %}
