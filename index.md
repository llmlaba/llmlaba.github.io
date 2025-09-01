---
layout: home
title: "Main page"
---

# Articles
- Occasionally, I write about LLM AI technologies that interest me on this blog.
- If you want to begin your learning journey about LLM AI, this blog will be very helpful for you.

## General

{% for article in site.articles %}
  {% if article.categories contains "general" %}
- [{{ article.title }}]({{ article.url | relative_url }}) — {{ article.date | date: "%d.%m.%Y" }}
  {% endif %}
{% endfor %}

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

## Known issues / errors
{% for article in site.articles %}
  {% if article.categories contains "issue" %}
- [{{ article.title }}]({{ article.url | relative_url }}) — {{ article.date | date: "%d.%m.%Y" }}
  {% endif %}
{% endfor %}
