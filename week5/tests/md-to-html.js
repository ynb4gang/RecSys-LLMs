#!/usr/bin/env node
/**
 * Minimal, deterministic Markdown -> HTML converter for week5/report.md.
 *
 * Supports exactly the subset used by the report: ATX headings (#..####),
 * paragraphs, unordered/ordered lists, pipe tables, blockquotes, fenced code
 * blocks, horizontal rules, bold, italic, inline code, markdown links and bare
 * http(s) autolinks. Output is a self-contained HTML document (inline CSS) that
 * headless Chrome can print to PDF:
 *
 *   node md-to-html.js ../report.md /tmp/report.html
 *   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
 *     --headless --disable-gpu --no-pdf-header-footer \
 *     --print-to-pdf=../report.pdf file:///tmp/report.html
 */
"use strict";

const fs = require("fs");

const CSS = `
@page { size: A4; margin: 20mm 17mm; }
* { box-sizing: border-box; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.5; color: #111;
  max-width: 100%; margin: 0;
}
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 6px; text-align: center; }
h2 { font-size: 13.5pt; margin: 22px 0 8px; border-bottom: 1px solid #444; padding-bottom: 3px; }
h3 { font-size: 11.5pt; margin: 16px 0 6px; }
h4 { font-size: 10.5pt; margin: 12px 0 4px; }
p  { margin: 8px 0; text-align: justify; }
.titleblock { text-align: center; margin-bottom: 14px; }
.titleblock p { text-align: center; margin: 2px 0; }
hr { border: 0; border-top: 1px solid #999; margin: 16px 0; }
code { font-family: Menlo, Consolas, monospace; font-size: 9pt;
       background: #f2f2f2; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px 10px;
      font-size: 9pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
a { color: #123a7a; text-decoration: none; word-break: break-all; }
blockquote { margin: 10px 0; padding: 4px 14px; border-left: 3px solid #666;
             background: #f7f7f7; font-style: italic; }
ul, ol { margin: 8px 0 8px 22px; padding: 0; }
li { margin: 3px 0; text-align: justify; }
table { border-collapse: collapse; margin: 10px 0; width: 100%;
        font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #888; padding: 4px 7px; text-align: left;
         vertical-align: top; }
th { background: #ececec; font-weight: bold; }
td.num, th.num { text-align: right; }
strong { font-weight: 700; }
.em { font-style: italic; }
.small { color: #333; }
`.trim();

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(text) {
  let t = escapeHtml(text);

  // 1. inline code spans (protect their contents from all later rules)
  const codeSpans = [];
  t = t.replace(/`([^`]+)`/g, (_, c) => {
    codeSpans.push(c);
    return `\u0000C${codeSpans.length - 1}\u0000`;
  });

  // 2. markdown links [text](url)
  t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) =>
    `<a href="${url}">${label}</a>`);

  // 3. bold, then italic (asterisk and underscore variants)
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/(^|[^*\w])\*([^*\n]+)\*(?=[^*\w]|$)/g, "$1<em>$2</em>");
  t = t.replace(/(^|[^_\w])_([^_\n]+)_(?=[^_\w]|$)/g, "$1<em>$2</em>");

  // 4. bare http(s) autolinks (code spans already protected)
  t = t.replace(/https?:\/\/[^\s<>"')\]]+/g, (m) => {
    const trail = m.match(/[.,;:]+$/);
    const core = trail ? m.slice(0, -trail[0].length) : m;
    return `<a href="${core}">${core}</a>${trail ? trail[0] : ""}`;
  });

  // 5. restore code spans
  t = t.replace(/\u0000C(\d+)\u0000/g, (_, i) => `<code>${codeSpans[+i]}</code>`);
  return t;
}

function splitRow(line) {
  const s = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return s.split("|").map((c) => c.trim());
}

function convert(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  let para = [];
  let inTitleBlock = false; // paragraphs directly after the h1 are centered

  const flushPara = () => {
    if (para.length) {
      out.push(inTitleBlock
        ? `<p class="titleblock">${inline(para.join(" "))}</p>`
        : `<p>${inline(para.join(" "))}</p>`);
      para = [];
    }
  };

  const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isSepRow = (l) => /^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$/.test(l);

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // blank
    if (trimmed === "") { flushPara(); i++; continue; }

    // horizontal rule
    if (/^---+$/.test(trimmed)) { flushPara(); inTitleBlock = false; out.push("<hr>"); i++; continue; }

    // headings
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushPara();
      const level = h[1].length;
      const cls = level === 1 ? ' class="titleblock"' : "";
      inTitleBlock = level === 1;
      out.push(`<h${level}${cls}>${inline(h[2])}</h${level}>`);
      i++; continue;
    }

    // fenced code block
    if (/^```/.test(trimmed)) {
      flushPara();
      inTitleBlock = false;
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i].trim())) { buf.push(lines[i]); i++; }
      i++; // closing fence
      out.push(`<pre><code>${escapeHtml(buf.join("\n"))}</code></pre>`);
      continue;
    }

    // blockquote
    if (/^>\s?/.test(trimmed)) {
      flushPara();
      inTitleBlock = false;
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i].trim())) {
        buf.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      out.push(`<blockquote>${inline(buf.join(" "))}</blockquote>`);
      continue;
    }

    // pipe table
    if (isTableRow(line) && i + 1 < lines.length && isSepRow(lines[i + 1])) {
      flushPara();
      inTitleBlock = false;
      const header = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && isTableRow(lines[i])) { rows.push(splitRow(lines[i])); i++; }
      out.push("<table><thead><tr>" +
        header.map((c) => `<th>${inline(c)}</th>`).join("") +
        "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>");
      continue;
    }

    // unordered list
    if (/^[-*]\s+/.test(trimmed)) {
      flushPara();
      inTitleBlock = false;
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i++;
      }
      out.push("<ul>" + items.map((x) => `<li>${inline(x)}</li>`).join("") + "</ul>");
      continue;
    }

    // ordered list
    if (/^\d+\.\s+/.test(trimmed)) {
      flushPara();
      inTitleBlock = false;
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i++;
      }
      out.push("<ol>" + items.map((x) => `<li>${inline(x)}</li>`).join("") + "</ol>");
      continue;
    }

    // paragraph continuation
    para.push(trimmed);
    i++;
  }
  flushPara();
  return out.join("\n");
}

// --------------------------------------------------------------------------- //
const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error("usage: node md-to-html.js <in.md> <out.html>");
  process.exit(1);
}
const md = fs.readFileSync(inPath, "utf8");
const titleMatch = md.match(/^#\s+(.+)$/m);
const title = titleMatch ? titleMatch[1] : "Report";
const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<style>${CSS}</style>
</head>
<body>
${convert(md)}
</body>
</html>
`;
fs.writeFileSync(outPath, html);
console.log(`wrote ${outPath} (${html.length} bytes)`);
