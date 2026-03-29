/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './_layouts/**/*.html',
    './_includes/**/*.html',
    './_articles/**/*.md',
    './_news/**/*.md',
    './*.md',
    './*.html',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          900: '#0d0d0d',
          800: '#141414',
          700: '#1c1c1c',
          600: '#242424',
        },
        accent: '#e05252',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
