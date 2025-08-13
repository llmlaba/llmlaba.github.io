---
layout: home
title: "Main page"
---

# Articles
Occasionally, I write about LLM AI technologies that interest me on this blog.  

{% for article in site.articles %}
- [{{ article.title }}]({{ article.url | relative_url }}) — {{ article.date | date: "%d.%m.%Y" }}
{% endfor %}
