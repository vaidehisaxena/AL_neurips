import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const outputDir = `${root}/outputs/neurips_draft`;
const data = JSON.parse(await fs.readFile(`${outputDir}/prism_workbook_data.json`, "utf8"));
const workbook = Workbook.create();
const colors = {
  phlox: "#CAA9F3",
  verbena: "#B37AD4",
  periwinkle: "#7997E6",
  atlantis: "#206ABC",
  phthalo: "#0E155E",
  pale: "#F4F6FD",
  border: "#D9D9D9",
  text: "#171A21",
};

function colLetter(n) {
  let s = "";
  while (n > 0) {
    n -= 1;
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26);
  }
  return s;
}

let sheetIndex = 0;
for (const [name, rows] of Object.entries(data)) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.tabColor = sheetIndex === 0 ? colors.phthalo : sheetIndex < 4 ? colors.atlantis : colors.periwinkle;
  const rowCount = rows.length;
  const colCount = rows[0].length;
  const endCol = colLetter(colCount);
  sheet.getRange("A1").write(rows);
  const used = sheet.getRange(`A1:${endCol}${rowCount}`);
  used.format.font = { name: "Arial", size: 10, color: colors.text };
  used.format.verticalAlignment = "center";
  used.format.borders = { preset: "inside", style: "thin", color: colors.border };
  const header = sheet.getRange(`A1:${endCol}1`);
  header.format = {
    fill: colors.phthalo,
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  header.format.rowHeight = 32;
  used.format.autofitColumns();
  for (let c = 0; c < colCount; c += 1) {
    const column = sheet.getRange(`${colLetter(c + 1)}1:${colLetter(c + 1)}${rowCount}`);
    if (column.format.columnWidth > 24) column.format.columnWidth = 24;
    if (column.format.columnWidth < 10) column.format.columnWidth = 10;
  }
  if (name === "Read me") {
    sheet.getRange(`B2:B${rowCount}`).format.wrapText = true;
    sheet.getRange(`B1:B${rowCount}`).format.columnWidth = 78;
    sheet.getRange(`A2:B${rowCount}`).format.autofitRows();
  }
  for (let r = 2; r <= rowCount; r += 2) {
    sheet.getRange(`A${r}:${endCol}${r}`).format.fill = colors.pale;
  }
  const headers = rows[0].map((v) => String(v));
  for (let c = 0; c < headers.length; c += 1) {
    const h = headers[c].toLowerCase();
    if (h.includes("recall") || h.includes("rate") || h.includes("fraction") || h.includes("difference") || h.includes("ci_")) {
      sheet.getRange(`${colLetter(c + 1)}2:${colLetter(c + 1)}${rowCount}`).format.numberFormat = "0.000";
    }
    if (h === "seed" || h === "cycle" || h.includes("decisions") || h.includes("claims") || h.includes("scaffolds") || h.includes("selections")) {
      sheet.getRange(`${colLetter(c + 1)}2:${colLetter(c + 1)}${rowCount}`).format.numberFormat = "0";
    }
  }
  if (rowCount > 30) sheet.freezePanes.freezeRows(1);
  sheetIndex += 1;
}

workbook.recalculate();
const previews = `${outputDir}/workbook_previews`;
await fs.mkdir(previews, { recursive: true });
for (const [name, rows] of Object.entries(data)) {
  const colCount = rows[0].length;
  const previewRows = Math.min(rows.length, 32);
  const preview = await workbook.render({
    sheetName: name,
    range: `A1:${colLetter(colCount)}${previewRows}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(`${previews}/${name.replaceAll(" ", "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const keyCheck = await workbook.inspect({
  kind: "table",
  range: "Core endpoint!A1:F12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 8,
});
await fs.writeFile(`${outputDir}/workbook_inspect.ndjson`, keyCheck.ndjson, "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(`${outputDir}/workbook_errors.ndjson`, errors.ndjson, "utf8");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/Prism_Ready_Results.xlsx`);
console.log(`${outputDir}/Prism_Ready_Results.xlsx`);
