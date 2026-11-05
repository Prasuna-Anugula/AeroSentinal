/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:         '#070B16',
        'bg-sec':   '#0B1220',
        surface:    'rgba(17,27,47,0.68)',
        primary:    '#69C9FF',
        secondary:  '#8B83FF',
        accent:     '#36E0C1',
        success:    '#39D98A',
        warning:    '#FFBF69',
        danger:     '#FF647C',
        'text-pri': '#F4F7FF',
        'text-sec': '#9AAAC2',
        border:     'rgba(148,180,220,0.16)',
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        body:    ['Inter', 'sans-serif'],
        mono:    ['"JetBrains Mono"', 'monospace'],
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(rgba(105,201,255,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(105,201,255,0.04) 1px,transparent 1px)',
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
    },
  },
  plugins: [],
}
