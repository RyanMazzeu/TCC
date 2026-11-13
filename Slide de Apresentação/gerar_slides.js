// Slides de Resultados e Conclusão (continuação do "Slide TCC .pdf", que vai até a metodologia).
// Mesmo visual do deck feito no Canva: fundo #F3F7F8, cartões brancos, verde-azulado #0F766E, âmbar #F2A93B,
// títulos em Cambria e texto em Calibri, numa tela de 20 × 11,25 pol. (1920 × 1080 px, o tamanho do Canva).
// As figuras vêm de Pesquisa/scripts/gerar_figuras_slides.py; os números, das tabelas do artigo.
//
// Uso: npm install pptxgenjs@3.12.0 && node gerar_slides.js   (gera "Slides - Resultados e Conclusão.pptx")

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const PASTA = __dirname;
const FIG = (nome) => path.join(PASTA, "figuras", nome);
const SAIDA = path.join(PASTA, "Slides - Resultados e Conclusão.pptx");
const PRIMEIRA_PAGINA = 14; // o deck da metodologia termina na página 13

const COR = {
  fundo: "F3F7F8", branco: "FFFFFF", teal: "0F766E", tealEscuro: "0F5757", tealClaro: "D5EBE7",
  ambar: "F2A93B", ambarClaro: "FCEBD0", titulo: "12263A", texto: "0F1F2E", cinza: "5B6B76", borda: "D7DBDD",
};
const COR_METODO = {
  "Sem RAG": "FFFFFF", Stuff: "2A78D6", Refine: "EDA100", "Map-Reduce": "E87BA4",
  "Map-Rerank": "008300", "Step-Down": "6250D6", Reciprocal: "1BAF7A",
};
const W = 20, H = 11.25, MX = 1.06, CW = W - 2 * MX; // margem lateral e largura útil

const pres = new pptxgen();
pres.defineLayout({ name: "TELA_CANVA", width: W, height: H });
pres.layout = "TELA_CANVA";
pres.title = "Resultados e conclusão";
pres.subject = "TCC: análise comparativa de métodos de RAG";
pres.author = "Afonso Henriques Massunari e Ryan Alves Mazzeu";
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };

// ---------- layouts ----------
function layoutConteudo(nome, secao) {
  pres.defineSlideMaster({
    title: nome,
    background: { color: COR.fundo },
    objects: [
      { text: { text: secao, options: { x: MX, y: 0.55, w: 12, h: 0.4, fontFace: "Calibri", fontSize: 14, bold: true,
        color: COR.teal, charSpacing: 6, margin: 0, valign: "middle" } } },
      { placeholder: { options: { name: "titulo", type: "title", x: MX, y: 0.98, w: CW, h: 1.05, fontFace: "Cambria",
        fontSize: 44, bold: true, color: COR.titulo, margin: 0, valign: "middle" }, text: "" } },
    ],
  });
}
layoutConteudo("RESULTADOS", "RESULTADOS");
layoutConteudo("CONCLUSAO", "CONCLUSÃO");
layoutConteudo("APENDICE", "APÊNDICE");
pres.defineSlideMaster({ title: "ENCERRAMENTO", background: { color: COR.branco }, objects: [] });

// ---------- utilidades ----------
function tamanhoPng(arquivo) {
  const b = fs.readFileSync(arquivo);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let pagina = PRIMEIRA_PAGINA;
function novoSlide(layout, secao, titulo, notas) {
  const s = pres.addSlide({ masterName: layout, sectionTitle: secao });
  if (titulo) s.addText(titulo, { placeholder: "titulo" });
  s.addText(String(pagina++), { x: W - MX - 1.2, y: H - 0.8, w: 1.2, h: 0.4, fontFace: "Calibri", fontSize: 14,
    color: COR.cinza, align: "right", margin: 0, isTextBox: true, objectName: "numero da página" });
  if (notas) s.addNotes(notas);
  return s;
}
function texto(s, t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: o.face || "Calibri", fontSize: o.size || 20, bold: !!o.bold,
    italic: !!o.italic, color: o.color || COR.texto, align: o.align || "left", valign: o.valign || "top",
    margin: o.margin ?? 0, isTextBox: true, paraSpaceAfter: o.paraSpaceAfter, charSpacing: o.charSpacing,
    lineSpacingMultiple: o.lineSpacingMultiple, objectName: o.nome });
}
function cartao(s, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: o.raio ?? 0.14,
    fill: { color: o.fill || COR.branco }, line: o.line ? { color: o.line, width: o.lineW || 1 } : { type: "none" },
    objectName: o.nome || "cartão" });
}
function faixa(s, t, y, h, o = {}) { // caixa verde-azulada com a mensagem do slide, como no deck da metodologia
  cartao(s, MX, y, CW, h, { fill: COR.teal, line: COR.tealEscuro, lineW: 1, nome: "faixa de mensagem" });
  texto(s, t, MX + 0.45, y, CW - 0.9, h, { size: o.size || 22, color: COR.branco, valign: "middle", nome: "mensagem" });
}
function subtitulo(s, t, y = 2.12) {
  texto(s, t, MX, y, CW, 0.5, { size: 21, color: COR.cinza, nome: "subtítulo" });
}
function pilula(s, t, x, y, w, h, o = {}) {
  cartao(s, x, y, w, h, { fill: o.fill || COR.tealClaro, raio: h / 2, nome: "rótulo" });
  texto(s, t, x, y, w, h, { size: o.size || 16, bold: true, color: o.color || COR.teal, align: "center",
    valign: "middle", nome: "texto do rótulo" });
}
function circulo(s, t, x, y, d, o = {}) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: o.fill || COR.ambar }, line: { type: "none" },
    objectName: "marcador" });
  texto(s, t, x, y, d, d, { size: o.size || 18, bold: true, color: o.color || COR.texto, align: "center",
    valign: "middle", nome: "texto do marcador" });
}
function estatistica(s, x, y, w, numero, rotulo, o = {}) {
  texto(s, numero, x, y, w, o.hNum || 0.85, { face: "Cambria", size: o.sizeNum || 44, bold: true,
    color: o.corNum || COR.teal, valign: "bottom", nome: "número" });
  texto(s, rotulo, x, y + (o.hNum || 0.85) + 0.08, w, o.hRot || 0.85, { size: o.sizeRot || 18,
    color: o.corRot || COR.cinza, nome: "legenda do número" });
}
function figura(s, arquivo, y, largura, o = {}) { // centralizada, com a proporção do PNG
  const t = tamanhoPng(FIG(arquivo));
  const h = largura * t.h / t.w;
  const x = o.x ?? (W - largura) / 2;
  s.addImage({ path: FIG(arquivo), x, y, w: largura, h, altText: o.alt || arquivo, objectName: "figura" });
  return y + h;
}
function linhaDeCartoes(s, itens, y, h, o = {}) { // cartões de mesma largura lado a lado, com número e legenda
  const gap = 0.21, n = itens.length, w = (CW - gap * (n - 1)) / n;
  itens.forEach(([numero, rotulo], i) => {
    const x = MX + i * (w + gap);
    cartao(s, x, y, w, h);
    estatistica(s, x + 0.35, y + 0.12, w - 0.7, numero, rotulo, { sizeNum: o.sizeNum || 38, hNum: 0.75,
      sizeRot: o.sizeRot || 17, hRot: h - 1.0 });
  });
}
function colunas(n, gap = 0.21) {
  const w = (CW - gap * (n - 1)) / n;
  return Array.from({ length: n }, (_, i) => ({ x: MX + i * (w + gap), w }));
}

// =====================================================================================================
// RESULTADOS
// =====================================================================================================
pres.addSection({ title: "Resultados" });

// 1. Visão geral
{
  const s = novoSlide("RESULTADOS", "Resultados", "O domínio decide o resultado",
    "Antes dos detalhes, a mensagem principal: o mesmo experimento produziu três comportamentos diferentes do RAG. " +
    "No FaQuAD, a recuperação multiplicou a qualidade; no MedQA, nenhum método mudou a acurácia; no Docugami, o RAG " +
    "ajudou, mas a qualidade ficou baixa porque a recuperação é o gargalo. Os próximos slides mostram cada um.");
  subtitulo(s, "O mesmo experimento produziu três comportamentos diferentes do RAG");
  const dados = [
    ["FaQuAD · normas de uma universidade", "1,7× a 2,5×", "a cobertura da resposta com o RAG, em todos os modelos",
      "O Stuff, o método mais simples e barato, foi o melhor."],
    ["MedQA · questões de medicina", "0", "métodos que mudaram a acurácia de forma significativa",
      "Os modelos já acertam de 71% a 96% sem RAG: o conteúdo já está neles."],
    ["Docugami · relatórios financeiros", "< 25%", "dos valores pedidos, mesmo no melhor caso dos econômicos",
      "A recuperação é o gargalo, e o Query Step-Down liderou."],
  ];
  colunas(3).forEach(({ x, w }, i) => {
    const [rotulo, numero, legenda, corpo] = dados[i];
    cartao(s, x, 3.0, w, 5.4);
    pilula(s, rotulo, x + 0.35, 3.35, w - 0.7, 0.55);
    texto(s, numero, x + 0.35, 4.15, w - 0.7, 1.05, { face: "Cambria", size: 56, bold: true, color: COR.teal,
      valign: "middle", nome: "número" });
    texto(s, legenda, x + 0.35, 5.3, w - 0.7, 0.9, { size: 19, color: COR.cinza });
    texto(s, corpo, x + 0.35, 6.45, w - 0.7, 1.6, { size: 20, color: COR.texto });
  });
  faixa(s, "A ordem dos métodos depende mais do domínio do que do LLM", 8.75, 1.05);
}

// 2. Recuperação
{
  const s = novoSlide("RESULTADOS", "Resultados", "Achar o documento certo não basta",
    "A qualidade da busca é a mesma para todos os LLMs, porque os fragmentos dependem só da pergunta. No FaQuAD, o " +
    "parágrafo certo veio em 91% das perguntas, e os fragmentos cobriram quase toda a resposta. No Docugami, o " +
    "relatório certo veio em 82% das perguntas, mas os fragmentos cobriram menos da metade da resposta de referência, " +
    "porque ela junta valores de vários lugares. No MedQA, o gabarito não indica o trecho, então só há o Context Recall.");
  subtitulo(s, "Qualidade dos quatro fragmentos que chegam ao LLM, a mesma para todos os modelos");
  const dados = [
    ["FaQuAD", "249 parágrafos · 900 perguntas", ["91%", "das perguntas com o parágrafo certo entre os 4 fragmentos"],
      ["0,94", "da resposta de referência coberta pelos fragmentos (Context Recall)"]],
    ["Docugami", "20 relatórios · 195 perguntas", ["82%", "das perguntas com o relatório certo entre os 4 fragmentos"],
      ["0,41", "da resposta coberta: menos da metade (Context Recall)"]],
    ["MedQA", "18 livros-texto · 121 mil fragmentos", ["—", "o gabarito não indica o trecho dos livros"],
      ["0,52", "da resposta coberta pelos livros-texto (Context Recall)"]],
  ];
  colunas(3).forEach(({ x, w }, i) => {
    const [nome, desc, a, b] = dados[i];
    cartao(s, x, 2.95, w, 5.5);
    texto(s, nome, x + 0.35, 3.25, w - 0.7, 0.5, { face: "Cambria", size: 26, bold: true, color: COR.titulo });
    texto(s, desc, x + 0.35, 3.8, w - 0.7, 0.4, { size: 16, color: COR.cinza });
    estatistica(s, x + 0.35, 4.35, w - 0.7, a[0], a[1], { hNum: 0.85, hRot: 0.9 });
    estatistica(s, x + 0.35, 6.25, w - 0.7, b[0], b[1], { hNum: 0.85, hRot: 0.9 });
  });
  faixa(s, "No Docugami, a resposta junta valores de tabelas, trimestres e empresas diferentes: quatro fragmentos " +
    "raramente trazem tudo", 8.75, 1.05, { size: 21 });
}

// 3. Efeito do RAG (figura)
{
  const s = novoSlide("RESULTADOS", "Resultados", "Efeito do RAG em cada domínio",
    "Cada linha é um LLM. O círculo vazado é o modelo sem RAG; o azul, o Stuff; o laranja, o melhor método de RAG " +
    "daquele modelo, com o nome à direita. No FaQuAD, o salto é grande em todos. No MedQA, os pontos quase se " +
    "sobrepõem. No Docugami, o RAG ajuda, mas tudo fica abaixo de 0,35. Os quadrados são os intermediários, que " +
    "responderam 25 perguntas.");
  const fim = figura(s, "efeito_rag.png", 2.3, 15.6, { alt: "Métrica principal sem RAG, com o Stuff e com o melhor " +
    "método de cada LLM, nos três conjuntos" });
  texto(s, "Círculos: econômicos (100 perguntas). Quadrados: intermediários (25 perguntas). Docugami: valores do " +
    "gabarito; MedQA: acurácia; FaQuAD: cobertura de tokens.", MX, fim + 0.15, CW, 0.45,
    { size: 16, color: COR.cinza, align: "center", nome: "legenda da figura" });
}

// 4. Quanto o RAG acrescentou
{
  const s = novoSlide("RESULTADOS", "Resultados", "Quanto o RAG acrescentou",
    "Os números dos cinco modelos econômicos. No FaQuAD, todos os seis métodos melhoraram todos os modelos, com " +
    "significância. No MedQA, o Stuff mudou a acurácia no máximo quatro pontos, e nenhum método mudou de forma " +
    "significativa depois da correção de Holm. No Docugami, o melhor método de cada modelo chegou a 13% a 24% dos " +
    "valores, com ganho significativo em quatro dos cinco, mas o Stuff ainda recusou de 42% a 71% das perguntas.");
  const dados = [
    ["FaQuAD", "cobertura da resposta", "sem RAG", "35–50%", "com o Stuff", "85–92%",
      ["Todos os 6 métodos melhoraram os 5 modelos", "Normas que o modelo não tem como saber de memória"]],
    ["MedQA", "acurácia", "sem RAG", "71–88%", "com o Stuff", "± 4 p.p.",
      ["Nenhum método mudou a acurácia de forma significativa", "Sonnet e Sol: 96% com ou sem RAG"]],
    ["Docugami", "valores do gabarito na resposta", "sem RAG", "0–7%", "melhor método", "13–24%",
      ["Ganho significativo em 4 dos 5 modelos", "Mesmo assim, o Stuff recusou 42% a 71%"]],
  ];
  colunas(3).forEach(({ x, w }, i) => {
    const [nome, metrica, r1, n1, r2, n2, itens] = dados[i];
    cartao(s, x, 2.55, w, 5.0);
    texto(s, nome, x + 0.35, 2.85, w - 0.7, 0.5, { face: "Cambria", size: 26, bold: true, color: COR.titulo });
    texto(s, metrica, x + 0.35, 3.38, w - 0.7, 0.4, { size: 16, color: COR.cinza });
    texto(s, r1, x + 0.35, 4.05, 2.25, 0.35, { size: 15, color: COR.cinza });
    texto(s, n1, x + 0.35, 4.4, 2.25, 0.75, { face: "Cambria", size: 34, bold: true, color: COR.cinza });
    texto(s, "→", x + 2.6, 4.4, 0.55, 0.75, { size: 34, bold: true, color: COR.ambar, align: "center" });
    texto(s, r2, x + 3.2, 4.05, w - 3.55, 0.35, { size: 15, color: COR.cinza });
    texto(s, n2, x + 3.2, 4.4, w - 3.55, 0.75, { face: "Cambria", size: 34, bold: true, color: COR.teal });
    s.addText(itens.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < itens.length - 1 } })),
      { x: x + 0.35, y: 5.6, w: w - 0.7, h: 1.8, fontFace: "Calibri", fontSize: 19, color: COR.texto, valign: "top",
        margin: 0, paraSpaceAfter: 10, isTextBox: true, objectName: "destaques" });
  });
  faixa(s, "Quanto mais o modelo já sabe, menor o espaço para o RAG", 7.9, 1.05);
}

// 5. Qual método venceu (tabela de postos)
{
  const s = novoSlide("RESULTADOS", "Resultados", "O melhor método muda com o domínio",
    "A tabela mostra o posto médio de cada método nos cinco econômicos: 1 é o melhor. No Docugami, o Query " +
    "Step-Down liderou, e o Map-Rerank ficou significativamente abaixo dele em todos os modelos. No FaQuAD, o Stuff " +
    "liderou, e o Reciprocal perdeu de 11 a 28 pontos para ele. No MedQA, não houve ordem consistente. Os dois " +
    "métodos que escolhem uma única resposta por nota de confiança ficaram entre os piores no Docugami e no FaQuAD.");
  const postos = [
    ["Stuff", "2,8", "1,4", "5,2"], ["Refine", "3,2", "3,0", "3,7"], ["Map-Reduce", "3,8", "2,8", "3,0"],
    ["Map-Rerank", "6,0", "5,0", "2,4"], ["Step-Down", "1,2", "3,0", "2,3"], ["Reciprocal", "4,0", "5,8", "4,4"],
  ];
  const melhor = { 1: "1,2", 2: "1,4", 3: "2,3" }, pior = { 1: "6,0", 2: "5,8", 3: "5,2" };
  const cab = { bold: true, color: COR.branco, fill: { color: COR.teal }, fontSize: 19, align: "center", valign: "middle" };
  const linhas = [[{ text: "Método", options: { ...cab, align: "left" } }, { text: "Docugami", options: cab },
    { text: "FaQuAD", options: cab }, { text: "MedQA", options: cab }]];
  postos.forEach((r) => linhas.push(r.map((v, j) => {
    const base = { fontSize: 19, valign: "middle", align: j ? "center" : "left", color: COR.texto, fill: { color: COR.branco } };
    if (j && v === melhor[j]) return { text: v, options: { ...base, bold: true, color: COR.branco, fill: { color: COR.teal } } };
    if (j && v === pior[j]) return { text: v, options: { ...base, bold: true, fill: { color: COR.ambar } } };
    return { text: v, options: base };
  })));
  s.addTable(linhas, { x: MX, y: 2.55, w: 9.6, colW: [3.0, 2.2, 2.2, 2.2], rowH: 0.62, fontFace: "Calibri",
    border: { type: "solid", color: COR.borda, pt: 1 }, margin: [0.04, 0.14, 0.04, 0.14], objectName: "tabela de postos" });
  texto(s, "Posto médio nos 5 modelos econômicos (1 = melhor). Verde-azulado: o melhor; âmbar: o pior.",
    MX, 7.05, 9.6, 0.75, { size: 16, color: COR.cinza, nome: "legenda da tabela" });
  const notas = [
    ["Docugami · Query Step-Down", "Maior proporção de valores; o Map-Rerank ficou significativamente abaixo dele nos 5 modelos"],
    ["FaQuAD · Stuff", "Maior cobertura com o método mais barato; o Reciprocal perdeu de 11 a 28 pontos para o Stuff"],
    ["MedQA · sem vencedor claro", "Sem ordem consistente entre os modelos (W = 0,39; p = 0,08): diferenças de poucos pontos"],
  ];
  notas.forEach(([cabecalho, corpo], i) => {
    const y = 2.55 + i * 2.13;
    cartao(s, 11.0, y, W - MX - 11.0, 1.92);
    texto(s, cabecalho, 11.35, y + 0.25, W - MX - 11.7, 0.45, { size: 21, bold: true, color: COR.teal });
    texto(s, corpo, 11.35, y + 0.78, W - MX - 11.7, 1.0, { size: 18, color: COR.texto });
  });
  faixa(s, "Map-Rerank e Reciprocal, que escolhem uma única resposta por nota de confiança, ficaram entre os piores " +
    "no Docugami e no FaQuAD", 9.05, 1.05, { size: 20 });
}

// 6. A ordem se repete entre os LLMs (mapa de calor)
{
  const s = novoSlide("RESULTADOS", "Resultados", "A ordem dos métodos se repete entre os LLMs",
    "Cada linha é um LLM, nas 25 perguntas que os sete responderam. A cor é a posição do método dentro daquele LLM: " +
    "quanto mais escuro, melhor. No Docugami e no FaQuAD, as colunas escuras são as mesmas em quase todos os " +
    "modelos, e o W de Kendall é alto. No MedQA, as cores se misturam, e os intermediários acertam 96% com quase " +
    "qualquer método: no teto da escala, a ordem não importa.");
  subtitulo(s, "Posição de cada método dentro de cada LLM nas 25 perguntas comuns (mais escuro = melhor)");
  figura(s, "ordem_metodos.png", 2.75, 14.4, { alt: "Mapa de calor com a métrica de cada método nos sete LLMs" });
  linhaDeCartoes(s, [
    ["W = 0,71", "Docugami: concordância alta entre os 5 econômicos (p = 0,003)"],
    ["W = 0,74", "FaQuAD: concordância alta entre os 5 econômicos (p = 0,002)"],
    ["W = 0,39", "MedQA: sem ordem consistente entre os 5 econômicos (p = 0,08)"],
  ], 8.35, 1.8, { sizeNum: 34, sizeRot: 17 });
}

// 7. Recusas no Docugami
{
  const s = novoSlide("RESULTADOS", "Resultados", "A maior parte das recusas é falha da recuperação",
    "Cada barra são as recusas de um LLM com um método, separadas pelo que o contexto tinha, segundo o juiz. Azul " +
    "escuro: a informação não chegou ao modelo; azul claro: chegou só em parte; laranja: estava toda lá, e o modelo " +
    "recusou assim mesmo. De 1.302 recusas, 72% foram por falta de informação. O Query Step-Down recusou muito menos " +
    "porque as perguntas alternativas trouxeram fragmentos que a busca original não achava.");
  figura(s, "recusas_docugami.png", 2.3, 15.4, { alt: "Recusas por método e LLM no Docugami, por conteúdo do contexto" });
  linhaDeCartoes(s, [
    ["72%", "das 1.302 recusas: sem a informação no contexto"],
    ["16%", "com só parte da informação no contexto"],
    ["13%", "injustificadas: o contexto tinha tudo"],
    ["90 × 323", "recusas do Query Step-Down × do Stuff"],
  ], 8.3, 1.85);
}

// 8. Sem a informação
{
  const s = novoSlide("RESULTADOS", "Resultados", "Recusar pouco não é sinal de qualidade",
    "Aqui estão só as respostas em que o contexto não tinha nada da informação. O prompt manda recusar; a parte " +
    "laranja é quando o modelo respondeu assim mesmo. Isso aconteceu em 46% dos casos, e 93% dessas respostas não " +
    "trouxeram nenhum valor correto. O Luna e o Sol, que quase não recusam quando a informação está lá, são os que " +
    "mais respondem sem ela: a taxa de recusa só diz algo quando lida junto com o que chegou ao contexto.");
  figura(s, "sem_informacao_docugami.png", 2.3, 15.4, { alt: "Respostas sem a informação no contexto, por método e " +
    "LLM, no Docugami: recusou ou respondeu" });
  linhaDeCartoes(s, [
    ["52%", "das respostas de RAG sem nada da informação no contexto"],
    ["46%", "delas foram respondidas assim mesmo"],
    ["93%", "dessas respostas sem nenhum valor correto"],
    ["74% · 65%", "Luna e Sol: os que menos recusam são os que mais respondem sem a informação"],
  ], 8.3, 1.85, { sizeRot: 16 });
}

// 9. Custo dos métodos
{
  const s = novoSlide("RESULTADOS", "Resultados", "Quanto cada método custa",
    "O custo relativo é o custo do método dividido pelo do Stuff, para cada LLM e conjunto, e a barra mostra a " +
    "mediana dos 15 pares. Map-Reduce e Map-Rerank custam quase o dobro; o Refine, 2,7 vezes; o Reciprocal e o " +
    "Query Step-Down, que geram e respondem quatro perguntas alternativas, mais de cinco vezes. O Step-Down também é " +
    "o mais lento.");
  const barras = [["Sem RAG", 0.4, "0,4×"], ["Stuff", 1.0, "1×"], ["Map-Reduce", 1.8, "1,8×"], ["Map-Rerank", 1.8, "1,8×"],
    ["Refine", 2.7, "2,7×"], ["Reciprocal", 5.3, "5,3×"], ["Step-Down", 5.7, "5,7×"]];
  const cx = MX, cy = 2.55, cw = 11.0, ch = 6.6;
  cartao(s, cx, cy, cw, ch);
  texto(s, "Custo em relação ao Stuff (mediana)", cx + 0.4, cy + 0.3, cw - 0.8, 0.5, { size: 22, bold: true,
    color: COR.titulo });
  const x0 = cx + 2.75, escala = 1.18, alt = 0.48;
  barras.forEach(([nome, v, rotulo], i) => {
    const y = cy + 1.1 + i * 0.75;
    texto(s, nome, cx + 0.4, y, 2.2, alt, { size: 19, color: COR.texto, align: "right", valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: x0, y, w: v * escala, h: alt, fill: { color: COR_METODO[nome] },
      line: nome === "Sem RAG" ? { color: COR.cinza, width: 1.25 } : { type: "none" }, objectName: `barra ${nome}` });
    texto(s, rotulo, x0 + v * escala + 0.15, y, 1.2, alt, { size: 19, bold: true, color: COR.texto, valign: "middle" });
  });
  const ladoX = cx + cw + 0.25, ladoW = W - MX - ladoX;
  [["2,9 s", "tempo mediano do Stuff, da recuperação ao fim da geração"],
    ["16,0 s", "tempo mediano do Query Step-Down, o mais lento"],
    ["6 chamadas", "ao LLM por pergunta no Query Step-Down; o Stuff faz 1"]].forEach(([n, r], i) => {
    const y = cy + i * 2.25;
    cartao(s, ladoX, y, ladoW, 2.05);
    estatistica(s, ladoX + 0.35, y + 0.15, ladoW - 0.7, n, r, { sizeNum: 40, hNum: 0.8, hRot: 0.9, sizeRot: 18 });
  });
  texto(s, "Custo relativo: custo do método ÷ custo do Stuff, por LLM e conjunto (mediana dos 15 pares da camada " +
    "principal). Tempo: mediana de 1.500 respostas por método.", MX, 9.4, CW, 0.75, { size: 16, color: COR.cinza,
    nome: "nota" });
}

// 10–12. Custo × qualidade, um conjunto por slide
[
  ["docugami", "Custo × qualidade no Docugami", "O Query Step-Down, o mais caro, teve a maior métrica em 5 dos 7 " +
    "modelos e empatou com o Refine nos outros 2",
    "Cada ponto é um método com um LLM: no eixo horizontal, o custo por mil perguntas, em escala logarítmica; no " +
    "vertical, a fração dos valores do gabarito. A cor é o método, e a letra, o modelo. No Docugami, o roxo, do Query " +
    "Step-Down, fica no alto e à direita em quase todos os modelos: aqui o método mais caro também foi o melhor."],
  ["faquad", "Custo × qualidade no FaQuAD", "O Stuff, o método de RAG mais barato, teve a maior cobertura ou " +
    "empatou com ela em 5 dos 7 modelos",
    "No FaQuAD acontece o oposto: o azul, do Stuff, fica no alto e à esquerda. Pagar por métodos mais complexos não " +
    "trouxe ganho quando a busca já acha o parágrafo certo."],
  ["medqa", "Custo × qualidade no MedQA", "Do método mais barato ao mais caro, o custo variou de 7 a 15 vezes, e a " +
    "acurácia, no máximo 12 pontos",
    "No MedQA, os pontos de cada modelo se espalham na horizontal e quase não sobem: mais custo não comprou acurácia. " +
    "Os intermediários, à direita, acertam 96% com quase qualquer método."],
].forEach(([conj, titulo, mensagem, notas]) => {
  const s = novoSlide("RESULTADOS", "Resultados", titulo, notas + " Marcadores que se sobreporiam foram afastados " +
    "sem inverter a ordem dos valores; uma linha fina liga cada um à posição real.");
  figura(s, `custo_desempenho_${conj}.png`, 2.25, 16.4, { alt: `Custo por mil perguntas e métrica principal de cada ` +
    `método e LLM no ${titulo.split(" no ")[1]}` });
  faixa(s, mensagem, 9.25, 0.95, { size: 21 });
});

// 13. Tokens × custo
{
  const s = novoSlide("RESULTADOS", "Resultados", "Tokens × custo: o preço do token pesa mais",
    "Mesmos pontos, com os tokens consumidos no eixo vertical. Dentro de cada modelo, os pontos formam uma diagonal, " +
    "porque o custo cresce na proporção dos tokens. A distância entre os modelos é o preço do token: com o mesmo " +
    "consumo, o Sonnet e o Sol custam 20 vezes o Haiku e o Luna. O MedQA e o FaQuAD têm o mesmo padrão (Figura 8 do " +
    "artigo).");
  figura(s, "custo_tokens_docugami.png", 2.25, 16.2, { alt: "Tokens por pergunta e custo por mil perguntas de cada " +
    "método e LLM no Docugami" });
  faixa(s, "Mesmo consumo, token 20 vezes mais caro: até o Stuff do Sonnet e do Sol custou mais que o Query " +
    "Step-Down do Haiku e do Luna", 9.2, 1.0, { size: 21 });
}

// 14. Econômico × intermediário
{
  const s = novoSlide("RESULTADOS", "Resultados", "O método certo vale um modelo mais caro",
    "A comparação é dentro de cada família: o modelo econômico com o seu melhor método contra o intermediário da mesma " +
    "empresa com o Stuff, nas mesmas 25 perguntas. No Docugami, o Haiku com o Query Step-Down trouxe 26% dos valores, " +
    "contra 10% do Sonnet com o Stuff, com significância, custando 3,2 vezes menos. O Luna teve vantagem parecida " +
    "sobre o Sol, sem significância. No MedQA e no FaQuAD, empataram, e o econômico custou de 7 a 20 vezes menos.");
  subtitulo(s, "Docugami, nas 25 perguntas que os dois modelos responderam");
  const linhas = [
    ["Haiku + Query Step-Down", "26%", "US$ 1,63", "Sonnet + Stuff", "10%", "US$ 5,23", "p = 0,018", "3,2× mais barato", true],
    ["Luna + Query Step-Down", "29%", "US$ 0,75", "Sol + Stuff", "16%", "US$ 2,47", "p = 0,12", "3,3× mais barato", false],
  ];
  linhas.forEach(([eNome, eVal, eCusto, iNome, iVal, iCusto, p, razao, sig], i) => {
    const y = 2.8 + i * 2.65, h = 2.4;
    const blocos = [[MX, eNome, eVal, eCusto, COR.teal], [8.25, iNome, iVal, iCusto, COR.cinza]];
    blocos.forEach(([x, nome, val, custo, cor]) => {
      cartao(s, x, y, 6.25, h);
      texto(s, nome, x + 0.35, y + 0.25, 5.6, 0.5, { size: 22, bold: true, color: COR.titulo });
      texto(s, val, x + 0.35, y + 0.85, 2.3, 1.0, { face: "Cambria", size: 50, bold: true, color: cor, valign: "middle" });
      texto(s, "dos valores pedidos", x + 2.7, y + 0.95, 3.2, 0.45, { size: 18, color: COR.cinza });
      texto(s, `${custo} por 1.000 perguntas`, x + 2.7, y + 1.4, 3.3, 0.45, { size: 18, color: COR.cinza });
    });
    circulo(s, "vs", 7.43, y + h / 2 - 0.4, 0.8, { size: 18 });
    const rx = 14.75, rw = W - MX - rx;
    cartao(s, rx, y, rw, h, { fill: sig ? COR.teal : COR.tealClaro });
    texto(s, p, rx + 0.3, y + 0.45, rw - 0.6, 0.7, { face: "Cambria", size: 32, bold: true,
      color: sig ? COR.branco : COR.teal, align: "center", valign: "middle" });
    texto(s, sig ? `significativo\n${razao}` : `não significativo\n${razao}`, rx + 0.3, y + 1.2, rw - 0.6, 0.9,
      { size: 18, color: sig ? COR.branco : COR.texto, align: "center" });
  });
  faixa(s, "MedQA e FaQuAD: empate, com o econômico de 7 a 20 vezes mais barato. Em nenhum conjunto o " +
    "intermediário foi significativamente superior.", 8.35, 1.15, { size: 21 });
}

// 15. A métrica muda a conclusão
{
  const s = novoSlide("RESULTADOS", "Resultados", "A métrica muda a conclusão",
    "No Docugami, as duas métricas contam histórias opostas. A similaridade de cosseno, usada no estudo de " +
    "referência, deu ao modelo sem RAG nota maior que ao Stuff em seis dos sete modelos, embora o Stuff trouxesse " +
    "mais valores corretos em todos. Respostas genéricas, sem nenhum número, ficam próximas da referência no espaço " +
    "de embeddings. Só com o cosseno, a conclusão seria que o RAG piora as respostas.");
  subtitulo(s, "Docugami: as duas métricas contam histórias opostas");
  const [esq, dir] = colunas(2, 0.3);
  cartao(s, esq.x, 2.8, esq.w, 3.3);
  texto(s, "Similaridade de cosseno", esq.x + 0.4, 3.05, esq.w - 0.8, 0.5, { size: 24, bold: true, color: COR.titulo });
  texto(s, "6 de 7", esq.x + 0.4, 3.6, esq.w - 0.8, 1.1, { face: "Cambria", size: 60, bold: true, color: COR.cinza,
    valign: "middle" });
  texto(s, "modelos deram nota maior ao sem RAG que ao Stuff (em 2 deles, por menos de 0,01)", esq.x + 0.4, 4.8,
    esq.w - 0.8, 1.0, { size: 19, color: COR.cinza });
  cartao(s, dir.x, 2.8, dir.w, 3.3, { fill: COR.teal });
  texto(s, "Valores do gabarito", dir.x + 0.4, 3.05, dir.w - 0.8, 0.5, { size: 24, bold: true, color: COR.branco });
  texto(s, "7 de 7", dir.x + 0.4, 3.6, dir.w - 0.8, 1.1, { face: "Cambria", size: 60, bold: true, color: COR.branco,
    valign: "middle" });
  texto(s, "modelos trouxeram mais valores corretos com o Stuff que sem RAG", dir.x + 0.4, 4.8, dir.w - 0.8, 1.0,
    { size: 19, color: COR.branco });
  [[esq, "ρ = 0,43", "correlação entre o cosseno e os valores corretos, em 3.850 respostas"],
    [dir, "0,70 × 0,86", "cosseno das respostas sem nenhum valor correto × com todos os valores"]].forEach(([c, n, r]) => {
    cartao(s, c.x, 6.35, c.w, 1.75);
    estatistica(s, c.x + 0.4, 6.45, c.w - 0.8, n, r, { sizeNum: 34, hNum: 0.7, hRot: 0.7, sizeRot: 18 });
  });
  faixa(s, "O estudo de referência mediu só o cosseno e recomendou o Reciprocal para documentos financeiros; aqui, " +
    "ele ficou entre os piores no Docugami", 8.4, 1.15, { size: 21 });
}

// 16. Robustez
{
  const s = novoSlide("RESULTADOS", "Resultados", "Os resultados resistem a dois testes",
    "Dois testes de robustez. À esquerda, a busca aproximada do ChromaDB contra a busca exata do FAISS, com os mesmos " +
    "vetores: nas duas bases menores, os mesmos quatro fragmentos em todas as consultas; no MedQA, em 82%. Onde " +
    "diferiram, o Stuff foi refeito com os fragmentos do FAISS, e a acurácia quase não mudou. À direita, a variação " +
    "entre três execuções do Stuff nos modelos sem temperatura 0, muito menor que as diferenças que sustentam as " +
    "conclusões.");
  const [esq, dir] = colunas(2, 0.3);
  const blocos = [
    [esq, "Busca aproximada × exata", "ChromaDB (HNSW) contra FAISS (busca exata), com os mesmos vetores", [
      ["100%", "mesmos 4 fragmentos no FaQuAD e no Docugami"],
      ["82%", "no MedQA, com 121 mil vetores"],
      ["0,68 → 0,69", "acurácia do Stuff refeito com o FAISS no MedQA (McNemar p = 1,0)"]]],
    [dir, "Variação entre execuções", "Stuff repetido 3 vezes nos 4 modelos sem temperatura 0", [
      ["≤ 0,053", "desvio-padrão médio da métrica entre as 3 execuções"],
      ["3 de 4", "modelos escolheram a mesma alternativa nas 3 execuções em todo o MedQA"],
      ["11 a 28 pontos", "perda do Reciprocal para o Stuff no FaQuAD: muito maior que a variação"]]],
  ];
  blocos.forEach(([c, cab, sub, itens]) => {
    cartao(s, c.x, 2.55, c.w, 7.45);
    texto(s, cab, c.x + 0.45, 2.85, c.w - 0.9, 0.55, { face: "Cambria", size: 26, bold: true, color: COR.titulo });
    texto(s, sub, c.x + 0.45, 3.45, c.w - 0.9, 0.45, { size: 17, color: COR.cinza });
    itens.forEach(([n, r], i) => estatistica(s, c.x + 0.45, 4.1 + i * 1.95, c.w - 0.9, n, r,
      { sizeNum: 36, hNum: 0.75, hRot: 0.85, sizeRot: 18 }));
  });
}

// =====================================================================================================
// CONCLUSÃO
// =====================================================================================================
pres.addSection({ title: "Conclusão" });

// 17. Perguntas de pesquisa
{
  const s = novoSlide("CONCLUSAO", "Conclusão", "Respostas às perguntas de pesquisa",
    "Voltando às quatro perguntas da introdução. A resposta curta para as três primeiras é a mesma: depende do " +
    "domínio, e o domínio pesa mais que o modelo. A quarta mostra que a escolha do banco vetorial não distorceu " +
    "os resultados.");
  const qps = [
    ["QP1", "Como os 6 métodos se comparam em qualidade e custo em cada domínio?",
      "Depende do domínio: Query Step-Down à frente no financeiro, Stuff no FaQuAD e nenhuma diferença relevante no MedQA."],
    ["QP2", "A ordem dos métodos se mantém entre LLMs diferentes?",
      "Sim no Docugami e no FaQuAD (W de Kendall de 0,71 e 0,74); no MedQA, não houve ordem consistente."],
    ["QP3", "Quanto o RAG acrescenta em relação ao LLM sem RAG?",
      "Muito quando o modelo não conhece o conteúdo (FaQuAD: 1,7 a 2,5 vezes); nada quando já conhece (MedQA)."],
    ["QP4", "A busca aproximada do banco vetorial altera os resultados?",
      "Não: os bancos entregaram os mesmos fragmentos em 82% a 100% das consultas, sem mudar a acurácia."],
  ];
  qps.forEach(([qp, pergunta, resposta], i) => {
    const y = 2.5 + i * 1.88, h = 1.68;
    cartao(s, MX, y, CW, h);
    circulo(s, qp, MX + 0.35, y + (h - 1.0) / 2, 1.0, { size: 19 });
    texto(s, pergunta, MX + 1.7, y + 0.2, CW - 2.0, 0.55, { size: 22, bold: true, color: COR.titulo, valign: "middle" });
    texto(s, resposta, MX + 1.7, y + 0.8, CW - 2.0, 0.7, { size: 20, color: COR.cinza });
  });
}

// 18. O que mais os dados mostraram
{
  const s = novoSlide("CONCLUSAO", "Conclusão", "O que mais os dados mostraram",
    "Três achados que vão além das perguntas de pesquisa e que valem para quem constrói sistemas de RAG: o método " +
    "pode substituir um modelo mais caro, a métrica de avaliação pode inverter a conclusão, e uma taxa de recusa " +
    "baixa pode esconder respostas sem sustentação.");
  const itens = [
    ["O método certo vale um modelo mais caro", "Os econômicos com o método adequado não ficaram significativamente " +
      "abaixo dos intermediários, a um custo de 3 a 20 vezes menor."],
    ["A métrica muda a conclusão", "Só com a similaridade de cosseno, o RAG pareceria piorar as respostas " +
      "financeiras, embora traga mais valores corretos."],
    ["Recusar pouco não é sinal de qualidade", "Os modelos que menos recusam com a informação no contexto são os " +
      "que mais respondem sem ela."],
  ];
  colunas(3).forEach(({ x, w }, i) => {
    const [cab, corpo] = itens[i];
    cartao(s, x, 2.55, w, 5.6);
    circulo(s, String(i + 1), x + 0.4, 2.95, 1.0, { fill: COR.teal, color: COR.branco, size: 26 });
    texto(s, cab, x + 0.4, 4.25, w - 0.8, 1.05, { face: "Cambria", size: 26, bold: true, color: COR.titulo });
    texto(s, corpo, x + 0.4, 5.45, w - 0.8, 2.4, { size: 20, color: COR.texto });
  });
  faixa(s, "A recuperação é o gargalo: quando a informação não chega ao modelo, trocar o LLM não resolve", 8.5, 1.05);
}

// 19. Recomendações
{
  const s = novoSlide("CONCLUSAO", "Conclusão", "Recomendações para quem constrói um RAG",
    "As recomendações práticas do trabalho, na ordem em que fazem sentido num projeto: medir o ponto de partida, " +
    "começar simples, só pagar por métodos complexos quando a recuperação for o problema, considerar um modelo " +
    "econômico com o método certo e avaliar com uma métrica que verifique o conteúdo.");
  const recs = [
    ["Medir o modelo sem RAG primeiro", "para saber se a recuperação acrescenta algo no domínio"],
    ["Começar pelo Stuff", "o mais barato em quase todas as combinações e o melhor quando a busca acha o trecho"],
    ["Reformular a pergunta só se a recuperação for o gargalo", "o Query Step-Down custa quase 6 vezes o Stuff"],
    ["Evitar escolher uma resposta por nota de confiança", "Map-Rerank e Reciprocal ficaram entre os piores no " +
      "Docugami e no FaQuAD"],
    ["Testar um modelo econômico com o método certo", "antes de pagar 20 vezes mais por token"],
    ["Medir o conteúdo, não só a semelhança", "a métrica deve verificar o valor, a alternativa ou o trecho pedido"],
  ];
  const [esq, dir] = colunas(2, 0.3);
  recs.forEach(([cab, det], i) => {
    const c = i < 3 ? esq : dir, y = 2.5 + (i % 3) * 2.45, h = 2.2;
    cartao(s, c.x, y, c.w, h);
    circulo(s, String(i + 1), c.x + 0.35, y + 0.35, 0.85, { size: 22 });
    texto(s, cab, c.x + 1.5, y + 0.3, c.w - 1.85, 0.8, { size: 22, bold: true, color: COR.titulo });
    texto(s, det, c.x + 1.5, y + 1.15, c.w - 1.85, 0.9, { size: 18, color: COR.cinza });
  });
}

// 20. Limitações e trabalhos futuros
{
  const s = novoSlide("CONCLUSAO", "Conclusão", "Limitações e trabalhos futuros",
    "As limitações mais importantes: amostra de 100 perguntas por conjunto, uma única configuração de recuperação, " +
    "métricas automáticas sem avaliação humana e quatro modelos sem temperatura 0. Os trabalhos futuros atacam " +
    "justamente esses pontos e a recuperação no domínio financeiro, que foi o gargalo.");
  const [esq, dir] = colunas(2, 0.3);
  const lim = ["100 perguntas por conjunto (25 nos modelos intermediários)",
    "Uma configuração fixa: BGE-M3, fragmentos de 1.000 caracteres e k = 4",
    "Juiz automático com AUROC de 0,70, sem avaliação humana",
    "4 dos 7 modelos sem temperatura 0; os outros métodos rodaram uma vez",
    "Modelos comerciais podem mudar sem aviso"];
  const fut = ["Variar k, o modelo de embedding e a compressão do contexto",
    "Testar rerankers e busca híbrida no domínio financeiro",
    "Incluir modelos de topo na camada de generalização",
    "Acrescentar conjuntos em português de outros domínios",
    "Validar as métricas automáticas com avaliação humana"];
  [[esq, "Limitações", lim, COR.branco, COR.titulo, COR.texto], [dir, "Trabalhos futuros", fut, COR.teal, COR.branco,
    COR.branco]].forEach(([c, cab, itens, fill, corCab, corTxt]) => {
    cartao(s, c.x, 2.55, c.w, 7.4, { fill, line: fill === COR.teal ? COR.tealEscuro : undefined });
    texto(s, cab, c.x + 0.5, 2.9, c.w - 1.0, 0.6, { face: "Cambria", size: 30, bold: true, color: corCab });
    s.addText(itens.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < itens.length - 1 } })),
      { x: c.x + 0.5, y: 3.8, w: c.w - 1.0, h: 5.8, fontFace: "Calibri", fontSize: 21, color: corTxt, valign: "top",
        margin: 0, paraSpaceAfter: 16, isTextBox: true, objectName: cab });
  });
}

// 21. Encerramento
{
  const s = novoSlide("ENCERRAMENTO", "Conclusão", null, "Agradecer e abrir para perguntas.");
  texto(s, "Obrigado!", MX, 3.0, 14.5, 1.6, { size: 80, bold: true, color: COR.teal, nome: "agradecimento" });
  texto(s, "Perguntas?", MX, 4.75, 14.5, 0.9, { size: 40, color: COR.titulo });
  texto(s, "Afonso Henriques Massunari | Ryan Alves Mazzeu", MX, 6.35, 14.5, 0.55, { size: 24, color: COR.texto });
  texto(s, "Orientador: Carlos Henrique Valerio de Moraes", MX, 6.95, 14.5, 0.55, { size: 24, color: COR.texto });
  texto(s, "UNIFEI · Novembro 2026", MX, 7.85, 14.5, 0.5, { size: 20, color: COR.teal });
  s.addImage({ path: FIG("logo_unifei.png"), x: 15.6, y: 6.55, w: 3.3, h: 3.3, altText: "Logotipo da UNIFEI",
    objectName: "logotipo" });
}

// =====================================================================================================
// APÊNDICE (slides de apoio para as perguntas)
// =====================================================================================================
pres.addSection({ title: "Apêndice" });

// A1. Recusas no FaQuAD
{
  const s = novoSlide("APENDICE", "Apêndice", "Recusas no FaQuAD",
    "No FaQuAD, com a busca achando o parágrafo certo em 91% das perguntas, o quadro se inverte: poucas recusas, mas " +
    "43% delas injustificadas. O Map-Rerank concentrou 22 das 51: ao responder com um fragmento por vez, basta que a " +
    "recusa de um fragmento sem a resposta receba a maior nota de confiança para o método recusar.");
  figura(s, "recusas_faquad.png", 2.3, 15.4, { alt: "Recusas por método e LLM no FaQuAD, por conteúdo do contexto" });
  linhaDeCartoes(s, [
    ["120", "recusas nos 7 modelos e 6 métodos"],
    ["43%", "injustificadas: o contexto tinha toda a informação"],
    ["22 de 51", "injustificadas vieram do Map-Rerank, mais que as suas recusas justificadas (14)"],
  ], 8.3, 1.85);
}

// A2. O que muda de um método para outro
{
  const s = novoSlide("APENDICE", "Apêndice", "O que muda de um método para outro",
    "A decomposição por método nos cinco econômicos. O Query Step-Down e o Reciprocal levam mais vezes a informação " +
    "completa ao modelo, o que explica a vantagem do Step-Down no Docugami; em compensação, o Step-Down quase não " +
    "recusa sem a informação. O Map-Rerank falha mesmo com a informação no contexto, e o Reciprocal perde na " +
    "geração, não na busca.");
  const itens = [
    ["32% × 20%", "Docugami: informação completa no contexto com o Query Step-Down e o Reciprocal × com os métodos " +
      "que usam a pergunta original"],
    ["23% × 81%", "Docugami: rejeição negativa do Query Step-Down × do Stuff, ou seja, o Step-Down quase não recusa " +
      "sem a informação"],
    ["45% · 8%", "Docugami: com toda a informação no contexto, o Map-Rerank recusou 45% e trouxe 8% dos valores " +
      "(Stuff: 42%)"],
    ["0,72 × 0,93", "FaQuAD: cobertura do Reciprocal × do Stuff com a informação completa no contexto; a perda vem " +
      "da geração"],
  ];
  const [esq, dir] = colunas(2, 0.3);
  itens.forEach(([n, r], i) => {
    const c = i % 2 ? dir : esq, y = 2.55 + Math.floor(i / 2) * 3.65;
    cartao(s, c.x, y, c.w, 3.4);
    estatistica(s, c.x + 0.45, y + 0.3, c.w - 0.9, n, r, { sizeNum: 44, hNum: 0.95, hRot: 1.7, sizeRot: 20,
      corRot: COR.texto });
  });
}

// A3. Comparação com o estudo de referência
{
  const s = novoSlide("APENDICE", "Apêndice", "Comparação com o estudo de referência",
    "Medianas da similaridade de cosseno, com o bge-small-en-v1.5, um dos três embeddings do estudo de referência. " +
    "No Docugami, o Reciprocal, primeiro lá, ficou em último aqui, e o Query Step-Down foi o primeiro. No MedQA, os " +
    "seis métodos ficaram a menos de 0,02 entre si, enquanto lá variaram de 0,148 a 0,678.");
  const dados = [
    ["Stuff", "0,895 (2º)", "0,838 (3º)", "0,641 (3º)", "0,757 (6º)"],
    ["Refine", "0,836 (5º)", "0,866 (2º)", "0,609 (4º)", "0,760 (5º)"],
    ["Map-Reduce", "0,873 (4º)", "0,836 (4º)", "0,569 (5º)", "0,769 (3º)"],
    ["Map-Rerank", "0,801 (6º)", "0,690 (5º)", "0,148 (6º)", "0,773 (2º)"],
    ["Step-Down", "0,890 (3º)", "0,883 (1º)", "0,678 (1º)", "0,769 (3º)"],
    ["Reciprocal", "0,971 (1º)", "0,683 (6º)", "0,673 (2º)", "0,776 (1º)"],
  ];
  const cab = { bold: true, color: COR.branco, fill: { color: COR.teal }, fontSize: 18, align: "center", valign: "middle" };
  const corpo = (v, j) => ({ text: v, options: { fontSize: 18, align: j ? "center" : "left", valign: "middle",
    color: COR.texto, fill: { color: COR.branco }, bold: j === 0 || (v.startsWith("0,971") || v.startsWith("0,683")) } });
  const linhas = [
    [{ text: "", options: { ...cab, rowspan: 2 } }, { text: "Docugami", options: { ...cab, colspan: 2 } },
      { text: "MedQA", options: { ...cab, colspan: 2 } }],
    [{ text: "Artigo-base", options: cab }, { text: "Este trabalho", options: cab },
      { text: "Artigo-base", options: cab }, { text: "Este trabalho", options: cab }],
    ...dados.map((r) => r.map(corpo)),
  ];
  s.addTable(linhas, { x: MX, y: 2.55, w: 12.4, colW: [2.8, 2.4, 2.4, 2.4, 2.4], rowH: 0.66, fontFace: "Calibri",
    border: { type: "solid", color: COR.borda, pt: 1 }, margin: [0.04, 0.14, 0.04, 0.14], objectName: "tabela do estudo de referência" });
  cartao(s, 13.75, 2.55, W - MX - 13.75, 5.3);
  texto(s, "O Reciprocal, melhor método do estudo de referência, ficou em último no Docugami, mesmo com a métrica " +
    "dele. No MedQA, os seis métodos ficaram a menos de 0,02 entre si.", 14.1, 2.9, W - MX - 14.45, 4.6,
    { size: 20, color: COR.texto });
  texto(s, "Mediana da similaridade de cosseno com a resposta de referência; entre parênteses, a posição do método.",
    MX, 8.05, 12.4, 0.6, { size: 16, color: COR.cinza });
}

// A4. RAGAS
{
  const s = novoSlide("APENDICE", "Apêndice", "Métricas RAGAS",
    "Leitura complementar, com o juiz gpt-4o-mini, validado em uma AUROC de 0,70. A Faithfulness mede se a resposta " +
    "se sustenta nos trechos e só vale para métodos com um conjunto único de trechos. No MedQA ela é baixa porque o " +
    "prompt autoriza o conhecimento próprio. A Answer Relevancy reproduz, no FaQuAD, a queda do Map-Rerank e do " +
    "Reciprocal.");
  const dados = [
    ["Sem RAG", "—", "—", "—", "0,28", "0,53", "0,56"], ["Stuff", "0,74", "0,22", "0,90", "0,28", "0,54", "0,69"],
    ["Refine", "0,77", "0,33", "0,86", "0,39", "0,51", "0,68"], ["Map-Reduce", "0,51", "0,19", "0,82", "0,35", "0,54", "0,77"],
    ["Map-Rerank", "0,65", "0,18", "0,85", "0,21", "0,50", "0,47"], ["Step-Down", "—", "—", "—", "0,44", "0,54", "0,68"],
    ["Reciprocal", "—", "—", "—", "0,29", "0,50", "0,50"],
  ];
  const cab = { bold: true, color: COR.branco, fill: { color: COR.teal }, fontSize: 18, align: "center", valign: "middle" };
  const linhas = [
    [{ text: "", options: { ...cab, rowspan: 2 } }, { text: "Faithfulness", options: { ...cab, colspan: 3 } },
      { text: "Answer Relevancy", options: { ...cab, colspan: 3 } }],
    ["Docugami", "MedQA", "FaQuAD", "Docugami", "MedQA", "FaQuAD"].map((t) => ({ text: t, options: cab })),
    ...dados.map((r) => r.map((v, j) => ({ text: v, options: { fontSize: 18, align: j ? "center" : "left",
      valign: "middle", color: COR.texto, fill: { color: COR.branco }, bold: j === 0 } }))),
  ];
  s.addTable(linhas, { x: MX, y: 2.55, w: 13.2, colW: [2.7, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75], rowH: 0.62,
    fontFace: "Calibri", border: { type: "solid", color: COR.borda, pt: 1 }, margin: [0.04, 0.14, 0.04, 0.14],
    objectName: "tabela RAGAS" });
  cartao(s, 14.55, 2.55, W - MX - 14.55, 5.6);
  estatistica(s, 14.9, 2.75, W - MX - 15.25, "0,70", "AUROC do juiz gpt-4o-mini no RAGBench: as métricas RAGAS " +
    "são uma leitura complementar", { sizeNum: 48, hNum: 0.95, hRot: 1.4, sizeRot: 18 });
  texto(s, "A Faithfulness só se aplica aos métodos com um conjunto único de trechos.", 14.9, 5.6, W - MX - 15.25,
    1.2, { size: 17, color: COR.cinza });
  texto(s, "Médias das respostas dos 7 LLMs.", MX, 8.35, 13.2, 0.5, { size: 16, color: COR.cinza });
}

(async () => {
  await pres.writeFile({ fileName: SAIDA });
  console.log("escrito:", SAIDA);
})();
