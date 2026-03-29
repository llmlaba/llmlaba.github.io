# LLM Laboratory

Jekyll-сайт на GitHub Pages со сборкой через Tailwind CSS + Webpack + GLightbox.

## Требования

- Ruby 3.x (`ruby-full`)
- Node.js 20+
- npm

Проверить:
```bash
ruby --version
node --version
npm --version
```

## Первый запуск

### 1. Установить Bundler локально (без sudo)

```bash
gem install bundler --user-install
```

После этого добавить путь в PATH. Найти его можно командой `gem env` — поле `USER INSTALLATION DIRECTORY`, добавить `/bin` к нему. Для удобства добавить в `~/.bashrc` или `~/.zshrc`:

```bash
export PATH="$HOME/.local/share/gem/ruby/3.2.0/bin:$PATH"
```

Применить:
```bash
source ~/.bashrc
```

### 2. Установить Ruby-зависимости в папку проекта

Ключевой момент — все gem-ы ставятся в `vendor/bundle/` внутри проекта, не в систему и не в домашнюю директорию. Это изолированная установка, не требует sudo и не засоряет систему.

```bash
bundle config set --local path 'vendor/bundle'
bundle install
```

Настройка `path` сохраняется в `.bundle/config` — bundler будет использовать её автоматически при каждом следующем запуске.

### 3. Установить Node-зависимости

```bash
npm install
```

## Разработка

Запустить Tailwind (watch) и Jekyll (livereload) параллельно:

```bash
export PATH="$HOME/.local/share/gem/ruby/3.2.0/bin:$PATH"
npm run dev
```

Jekyll слушает на `0.0.0.0` — сайт доступен как локально на `http://localhost:4000`, так и по сети на `http://<ip-машины>:4000`. Livereload использует дополнительный порт **35729** — оба порта должны быть открыты.

## Сборка для продакшена

```bash
export PATH="$HOME/.local/share/gem/ruby/3.2.0/bin:$PATH"
npm run build
```

Результат в папке `_site/`.

## Структура команд

| Команда | Что делает |
|---|---|
| `npm run dev` | Tailwind watch + Jekyll livereload |
| `npm run build` | Полная сборка (CSS + JS + Jekyll) |
| `npm run build:front` | Только Tailwind + Webpack |
| `npm run build:css` | Только Tailwind |
| `npm run build:js` | Только Webpack |

## Структура проекта

```
├── .github/workflows/deploy.yml   # GitHub Actions — автодеплой на push в main
├── _articles/                     # Статьи (Jekyll collection)
├── _news/                         # Новости (Jekyll collection)
├── _layouts/
│   ├── default.html               # Layout для статей (topbar + сайдбар)
│   └── home.html                  # Layout главной страницы (topbar + hero)
├── _includes/
│   ├── gallery.html               # Галерея через GLightbox
│   └── video.html                 # Видео-плеер
├── assets/
│   ├── css/main.css               # Генерируется (не коммитить)
│   └── js/bundle.js               # Генерируется (не коммитить)
├── src/
│   ├── css/input.css              # Tailwind + стили галереи
│   └── js/main.js                 # GLightbox + Copy button + навигация
├── vendor/bundle/                 # Ruby gems (не коммитить)
├── _config.yml
├── Gemfile
├── Gemfile.lock                   # Коммитить — фиксирует версии gems
├── package.json
├── package-lock.json              # Коммитить — фиксирует версии npm
├── tailwind.config.js
└── webpack.config.js
```

## Деплой

Деплой происходит автоматически при push в ветку `main` через GitHub Actions.

Перед первым деплоем убедиться что в настройках репозитория:
**Settings → Pages → Source → GitHub Actions**
