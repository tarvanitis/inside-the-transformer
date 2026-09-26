// tex2svg.js — batch LaTeX → self-contained SVG via MathJax (v3, full TeX).
// Reads JSON on stdin: [{ "tex": "...", "display": true }, ...]
// Writes JSON on stdout: ["<svg ...>...</svg>", ...]
// Used by book/build-pdf.py so equations render crisply in the WeasyPrint PDF
// (WeasyPrint's native MathML is too weak for fractions/matrices).

const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: 'local' }),
});

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => (input += d));
process.stdin.on('end', () => {
  const items = JSON.parse(input);
  const out = items.map((it) => {
    const node = doc.convert(it.tex, { display: !!it.display });
    return adaptor.innerHTML(node); // the <svg>…</svg> inside the mjx-container
  });
  process.stdout.write(JSON.stringify(out));
});
