const fs = require('fs');
const MarkdownIt = require('markdown-it');
const mathjax3 = require('markdown-it-mathjax3');
const puppeteer = require('puppeteer');

const md = new MarkdownIt({ html: true }).use(mathjax3);

const content = fs.readFileSync('research_paper.md', 'utf-8');
const htmlContent = md.render(content);

const html = `<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Research Paper</title>
    <style>
        body {
            font-family: 'Times New Roman', Times, serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
            padding: 2em;
            color: #333;
        }
        h1, h2, h3 {
            color: #111;
        }
        h1 { font-size: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; margin-top: 1.5em; }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th { background-color: #f4f4f4; }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 1em;
            border-radius: 4px;
            overflow-x: auto;
        }
        .math {
            overflow-x: auto;
        }
    </style>
</head>
<body>
    ${htmlContent}
</body>
</html>`;

(async () => {
    console.log('Launching browser...');
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    
    // Determine the absolute path for local files to resolve properly
    const path = require('path');
    const fileUrl = 'file://' + path.resolve('temp.html').replace(/\\/g, '/');
    
    fs.writeFileSync('temp.html', html);
    
    console.log('Loading content...');
    await page.goto(fileUrl, { waitUntil: 'networkidle0' });
    
    console.log('Generating PDF...');
    await page.pdf({ 
        path: 'research_paper_fixed.pdf', 
        format: 'A4',
        printBackground: true,
        margin: { top: '2cm', right: '2cm', bottom: '2cm', left: '2cm' }
    });
    
    await browser.close();
    fs.unlinkSync('temp.html');
    console.log('Done! PDF generated at research_paper_fixed.pdf');
})();
