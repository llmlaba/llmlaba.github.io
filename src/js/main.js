import '../css/input.css';
import GLightbox from 'glightbox';
import 'glightbox/dist/css/glightbox.min.css';

document.addEventListener('DOMContentLoaded', () => {
  // Галерея
  GLightbox({ selector: '.glightbox' });

  // Copy button для блоков кода
  document.querySelectorAll('pre code').forEach((block) => {
    const pre = block.parentNode;
    pre.style.position = 'relative';

    const btn = document.createElement('button');
    btn.textContent = 'Copy';
    btn.className = 'copy-btn';
    btn.setAttribute('aria-label', 'Copy code');

    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(block.textContent).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => (btn.textContent = 'Copy'), 2000);
      });
    });

    pre.appendChild(btn);
  });

  // Активный пункт в навигации
  const currentPath = window.location.pathname;
  document.querySelectorAll('#sidebar a').forEach((link) => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });
});
