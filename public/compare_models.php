<?php
require_once __DIR__ . '/../app/services/PredictionService.php';
require_login();

$user = current_user();
$allowedDatasets = ['B-dataset', 'C-dataset', 'F-dataset'];
$dataset = $_POST['dataset'] ?? 'C-dataset';
$dataset = in_array($dataset, $allowedDatasets, true) ? $dataset : 'C-dataset';
$topK = max(1, min(20, (int) ($_POST['top_k'] ?? 5)));
$drugInput = trim((string) ($_POST['drugs'] ?? ''));
$diseaseInput = trim((string) ($_POST['diseases'] ?? ''));
$resultData = null;
$error = null;
$apiHealthy = PredictionService::isApiHealthy();

if (empty($_SESSION['_csrf_compare_models'])) {
    $_SESSION['_csrf_compare_models'] = bin2hex(random_bytes(16));
}
$csrfToken = $_SESSION['_csrf_compare_models'];

function parse_compare_entities(string $raw): array
{
    $parts = preg_split('/[\r\n,;]+/', $raw) ?: [];
    $items = [];
    $seen = [];
    foreach ($parts as $part) {
        $value = trim($part);
        if ($value === '') {
            continue;
        }
        $key = strtolower($value);
        if (isset($seen[$key])) {
            continue;
        }
        $items[] = $value;
        $seen[$key] = true;
    }
    return $items;
}

function format_score(mixed $score): string
{
    return number_format((float) $score, 4);
}

function render_compare_graph(array $graph): string
{
    $nodes = $graph['nodes'] ?? [];
    $links = $graph['links'] ?? [];
    $drugs = [];
    $diseases = [];
    foreach ($nodes as $node) {
        if (($node['type'] ?? '') === 'drug') {
            $drugs[] = $node;
        } elseif (($node['type'] ?? '') === 'disease') {
            $diseases[] = $node;
        }
    }

    $rowCount = max(count($drugs), count($diseases), 1);
    $height = max(360, 110 + ($rowCount * 82));
    $leftX = 190;
    $rightX = 910;
    $topPad = 70;
    $bottomPad = 54;
    $usableHeight = $height - $topPad - $bottomPad;
    $drugPositions = [];
    $diseasePositions = [];

    foreach ($drugs as $index => $node) {
        $y = $topPad + (($index + 0.5) * $usableHeight / max(count($drugs), 1));
        $drugPositions[$node['id']] = [$leftX, $y, $node];
    }
    foreach ($diseases as $index => $node) {
        $y = $topPad + (($index + 0.5) * $usableHeight / max(count($diseases), 1));
        $diseasePositions[$node['id']] = [$rightX, $y, $node];
    }

    ob_start();
    ?>
    <svg class="compare-network" viewBox="0 0 1100 <?= e((string) $height) ?>" role="img" aria-label="So do lien ket 2D drug disease">
        <defs>
            <marker id="arrow-improved" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#198754"></path>
            </marker>
            <marker id="arrow-original" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#d97706"></path>
            </marker>
        </defs>
        <text x="110" y="34" class="compare-network-title">Thuoc</text>
        <text x="845" y="34" class="compare-network-title">Benh</text>
        <?php foreach ($links as $link): ?>
            <?php
            $source = $drugPositions[$link['source'] ?? ''] ?? null;
            $target = $diseasePositions[$link['target'] ?? ''] ?? null;
            if (!$source || !$target) {
                continue;
            }
            $delta = (float) ($link['delta'] ?? 0);
            $score = max(0.05, min(1.0, (float) ($link['improved_score'] ?? 0)));
            $stroke = $delta >= 0 ? '#198754' : '#d97706';
            $marker = $delta >= 0 ? 'url(#arrow-improved)' : 'url(#arrow-original)';
            $width = 1.5 + ($score * 5);
            $midX = ((float) $source[0] + (float) $target[0]) / 2;
            $midY = ((float) $source[1] + (float) $target[1]) / 2;
            ?>
            <path d="M <?= e((string) $source[0]) ?> <?= e((string) $source[1]) ?> C 410 <?= e((string) $source[1]) ?>, 690 <?= e((string) $target[1]) ?>, <?= e((string) $target[0]) ?> <?= e((string) $target[1]) ?>"
                  stroke="<?= e($stroke) ?>" stroke-width="<?= e(number_format($width, 2)) ?>" marker-end="<?= e($marker) ?>" class="compare-edge"></path>
            <g class="compare-edge-label">
                <rect x="<?= e((string) ($midX - 58)) ?>" y="<?= e((string) ($midY - 14)) ?>" width="116" height="28" rx="8"></rect>
                <text x="<?= e((string) $midX) ?>" y="<?= e((string) ($midY + 4)) ?>">I <?= e(format_score($link['improved_score'] ?? 0)) ?> / Δ <?= e(format_score($delta)) ?></text>
            </g>
        <?php endforeach; ?>

        <?php foreach ($drugPositions as $position): ?>
            <?php [$x, $y, $node] = $position; ?>
            <g class="compare-node compare-node-drug">
                <circle cx="<?= e((string) $x) ?>" cy="<?= e((string) $y) ?>" r="12"></circle>
                <rect x="<?= e((string) ($x - 150)) ?>" y="<?= e((string) ($y - 24)) ?>" width="132" height="48" rx="8"></rect>
                <text x="<?= e((string) ($x - 84)) ?>" y="<?= e((string) ($y - 3)) ?>"><?= e((string) ($node['label'] ?? $node['actual_id'] ?? 'Drug')) ?></text>
                <text x="<?= e((string) ($x - 84)) ?>" y="<?= e((string) ($y + 15)) ?>" class="compare-node-id"><?= e((string) ($node['actual_id'] ?? '')) ?></text>
            </g>
        <?php endforeach; ?>

        <?php foreach ($diseasePositions as $position): ?>
            <?php [$x, $y, $node] = $position; ?>
            <g class="compare-node compare-node-disease">
                <circle cx="<?= e((string) $x) ?>" cy="<?= e((string) $y) ?>" r="12"></circle>
                <rect x="<?= e((string) ($x + 18)) ?>" y="<?= e((string) ($y - 24)) ?>" width="150" height="48" rx="8"></rect>
                <text x="<?= e((string) ($x + 93)) ?>" y="<?= e((string) ($y - 3)) ?>"><?= e((string) ($node['label'] ?? $node['actual_id'] ?? 'Disease')) ?></text>
                <text x="<?= e((string) ($x + 93)) ?>" y="<?= e((string) ($y + 15)) ?>" class="compare-node-id"><?= e((string) ($node['actual_id'] ?? '')) ?></text>
            </g>
        <?php endforeach; ?>
    </svg>
    <?php
    return (string) ob_get_clean();
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $submittedToken = (string) ($_POST['csrf_token'] ?? '');
    $drugs = parse_compare_entities($drugInput);
    $diseases = parse_compare_entities($diseaseInput);

    if (!hash_equals($csrfToken, $submittedToken)) {
        $error = 'Phien lam viec khong hop le. Vui long tai lai trang.';
    } elseif (!$apiHealthy) {
        $error = 'Python API dang ngoai tuyen. Vui long khoi dong FastAPI o cong 8000.';
    } elseif (empty($drugs) || empty($diseases)) {
        $error = 'Vui long nhap it nhat 1 thuoc va 1 benh.';
    } elseif (count($drugs) > 5 || count($diseases) > 5) {
        $error = 'Chi duoc nhap toi da 5 thuoc va 5 benh.';
    } else {
        try {
            $resultData = PredictionService::comparePredict($drugs, $diseases, $topK, $dataset);
            PredictionService::saveComparisonHistory((int) $user['id'], $dataset, $drugs, $diseases, $topK, $resultData);
            flash('success', 'Da chay so sanh 2 mo hinh thanh cong');
        } catch (Throwable $e) {
            $error = $e->getMessage();
        }
    }
}

$success = flash('success');
$comparisonRows = $resultData['comparison'] ?? [];
$chartRows = array_slice($comparisonRows, 0, 10);
$chartLabels = array_map(
    fn ($row) => ($row['drug_id'] ?? '') . ' / ' . ($row['disease_id'] ?? ''),
    $chartRows
);
$chartOriginal = array_map(fn ($row) => (float) ($row['original_score'] ?? 0), $chartRows);
$chartImproved = array_map(fn ($row) => (float) ($row['improved_score'] ?? 0), $chartRows);
$chartDelta = array_map(fn ($row) => (float) ($row['delta'] ?? 0), $chartRows);
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>So sanh 2 mo hinh</title>
    <link rel="stylesheet" href="assets/style.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<div id="loader" class="loading-overlay">Dang chay 2 mo hinh tren cung tap dau vao...</div>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">So sanh 2 mo hinh</div>
            <div class="muted">Chay model goc va model cai tien tren cung danh sach thuoc - benh.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="index.php">Dashboard</a>
            <a class="btn btn-ghost" href="history.php">Lich su</a>
            <a class="btn btn-danger" href="logout.php">Dang xuat</a>
        </div>
    </div>

    <?php if (!$apiHealthy): ?>
        <div class="alert alert-error">Canh bao: Python API o cong 8000 dang ngat ket noi.</div>
    <?php endif; ?>
    <?php if ($error): ?>
        <div class="alert alert-error"><?= e($error) ?></div>
    <?php endif; ?>
    <?php if ($success): ?>
        <div class="alert alert-success"><?= e($success) ?></div>
    <?php endif; ?>

    <div class="glass-card search-container">
        <div class="section-header">
            <div>
                <h2>Chay chan doan 2 mo hinh</h2>
                <p class="muted">Nhap ma ID hoac ten. Co the tach bang dau phay hoac xuong dong.</p>
            </div>
            <span class="badge <?= $apiHealthy ? 'badge-success' : 'badge-neutral' ?>"><?= $apiHealthy ? 'API Online' : 'API Offline' ?></span>
        </div>

        <form method="post" id="compare-form" onsubmit="document.getElementById('loader').style.display='grid'">
            <input type="hidden" name="csrf_token" value="<?= e($csrfToken) ?>">
            <input type="hidden" name="drugs" id="drugs-hidden" value="<?= e($drugInput) ?>">
            <input type="hidden" name="diseases" id="diseases-hidden" value="<?= e($diseaseInput) ?>">
            <div class="compare-form-grid">
                <div class="form-group">
                    <label class="label">Dataset</label>
                    <select class="select" name="dataset" id="dataset-select">
                        <?php foreach ($allowedDatasets as $option): ?>
                            <option value="<?= e($option) ?>" <?= $dataset === $option ? 'selected' : '' ?>><?= e($option) ?></option>
                        <?php endforeach; ?>
                    </select>
                </div>
                <div class="form-group">
                    <label class="label">Top-K</label>
                    <input class="input" type="number" name="top_k" min="1" max="20" value="<?= e((string) $topK) ?>">
                </div>
                <div class="form-group compare-span-2">
                    <label class="label">Danh sach thuoc (toi da 5)</label>
                    <div class="entity-picker" id="drug-picker">
                        <div class="entity-picker-tags" id="drug-tags"></div>
                        <input type="text" class="entity-picker-search" id="drug-search" placeholder="Tim kiem thuoc..." autocomplete="off">
                        <div class="entity-picker-list" id="drug-list"><div class="entity-picker-msg">Dang tai...</div></div>
                    </div>
                </div>
                <div class="form-group compare-span-2">
                    <label class="label">Danh sach benh (toi da 5)</label>
                    <div class="entity-picker" id="disease-picker">
                        <div class="entity-picker-tags" id="disease-tags"></div>
                        <input type="text" class="entity-picker-search" id="disease-search" placeholder="Tim kiem benh..." autocomplete="off">
                        <div class="entity-picker-list" id="disease-list"><div class="entity-picker-msg">Dang tai...</div></div>
                    </div>
                </div>
                <div class="form-group compare-action">
                    <button class="btn btn-full" type="submit" <?= !$apiHealthy ? 'disabled' : '' ?>>Chay chan doan 2 mo hinh</button>
                </div>
            </div>
        </form>
    </div>

    <?php if ($resultData): ?>
        <div class="grid grid-2 section-spaced">
            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>Model goc</h3>
                        <p class="muted mono"><?= e((string) ($resultData['models']['original']['checkpoint'] ?? '')) ?></p>
                    </div>
                    <span class="badge badge-neutral">Original</span>
                </div>
                <div class="table-container">
                    <table class="table">
                        <thead><tr><th>Thuoc</th><th>Benh</th><th>Score</th></tr></thead>
                        <tbody>
                        <?php foreach (($resultData['models']['original']['results'] ?? []) as $row): ?>
                            <tr>
                                <td><strong><?= e((string) $row['drug_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['drug_id']) ?></span></td>
                                <td><strong><?= e((string) $row['disease_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['disease_id']) ?></span></td>
                                <td class="score-text"><?= e(format_score($row['score'] ?? 0)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>Model cai tien</h3>
                        <p class="muted mono"><?= e((string) ($resultData['models']['improved']['checkpoint'] ?? '')) ?></p>
                    </div>
                    <span class="badge badge-success">Improved</span>
                </div>
                <div class="table-container">
                    <table class="table">
                        <thead><tr><th>Thuoc</th><th>Benh</th><th>Score</th></tr></thead>
                        <tbody>
                        <?php foreach (($resultData['models']['improved']['results'] ?? []) as $row): ?>
                            <tr>
                                <td><strong><?= e((string) $row['drug_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['drug_id']) ?></span></td>
                                <td><strong><?= e((string) $row['disease_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['disease_id']) ?></span></td>
                                <td class="score-text"><?= e(format_score($row['score'] ?? 0)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="glass-card section-spaced">
            <div class="section-header">
                <div>
                    <h3>Bang so sanh delta</h3>
                    <p class="muted">Delta = improved_score - original_score.</p>
                </div>
                <span class="badge badge-drug"><?= count($comparisonRows) ?> cap</span>
            </div>
            <div class="table-container">
                <table class="table">
                    <thead>
                    <tr>
                        <th>Thuoc</th>
                        <th>Benh</th>
                        <th>Original</th>
                        <th>Improved</th>
                        <th>Delta</th>
                        <th>Winner</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($comparisonRows as $row): ?>
                        <tr>
                            <td><strong><?= e((string) $row['drug_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['drug_id']) ?></span></td>
                            <td><strong><?= e((string) $row['disease_name']) ?></strong><br><span class="muted mono"><?= e((string) $row['disease_id']) ?></span></td>
                            <td class="score-text"><?= e(format_score($row['original_score'] ?? 0)) ?></td>
                            <td class="score-text"><?= e(format_score($row['improved_score'] ?? 0)) ?></td>
                            <td class="<?= ((float) ($row['delta'] ?? 0)) >= 0 ? 'delta-positive' : 'delta-negative' ?>"><?= e(format_score($row['delta'] ?? 0)) ?></td>
                            <td><span class="badge <?= ($row['winner'] ?? '') === 'improved' ? 'badge-success' : 'badge-neutral' ?>"><?= e((string) ($row['winner'] ?? 'tie')) ?></span></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="grid grid-2 section-spaced compare-visual-grid">
            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>Bieu do cot</h3>
                        <p class="muted">Hien thi toi da 10 cap theo thu tu improved_score.</p>
                    </div>
                </div>
                <div class="chart-panel">
                    <canvas id="comparisonChart"></canvas>
                    <div id="chartFallback" class="chart-fallback">Khong tai duoc Chart.js CDN. Vui long kiem tra ket noi mang hoac xem bang delta ben tren.</div>
                </div>
            </div>
            <div class="glass-card">
                <div class="section-header">
                    <div>
                        <h3>So do lien ket 2D</h3>
                        <p class="muted">Cot trai la thuoc, cot phai la benh, canh noi la ket qua du doan.</p>
                    </div>
                </div>
                <?= render_compare_graph($resultData['graph2d'] ?? []) ?>
            </div>
        </div>
    <?php endif; ?>
</div>

<?php if ($resultData): ?>
<script>
const chartLabels = <?= json_encode($chartLabels, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const originalScores = <?= json_encode($chartOriginal, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const improvedScores = <?= json_encode($chartImproved, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const deltas = <?= json_encode($chartDelta, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const chartElement = document.getElementById('comparisonChart');
if (chartElement && window.Chart) {
    new Chart(chartElement, {
        type: 'bar',
        data: {
            labels: chartLabels,
            datasets: [
                { label: 'Original model score', data: originalScores, backgroundColor: 'rgba(126, 108, 79, 0.55)' },
                { label: 'Improved model score', data: improvedScores, backgroundColor: 'rgba(31, 94, 255, 0.72)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        afterBody: (items) => {
                            const index = items[0]?.dataIndex ?? 0;
                            return `Delta: ${Number(deltas[index] || 0).toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                y: { beginAtZero: true, max: 1 },
                x: { ticks: { maxRotation: 45, minRotation: 0 } }
            }
        }
    });
} else {
    const fallback = document.getElementById('chartFallback');
    if (fallback) {
        fallback.style.display = 'grid';
    }
}
</script>
<?php endif; ?>

<script>
(() => {
    const API = <?= json_encode(rtrim((string) config('python_api.base_url'), '/'), JSON_UNESCAPED_SLASHES) ?>;
    const MAX = 5;
    const preselDrugs = <?= json_encode(parse_compare_entities($drugInput), JSON_UNESCAPED_UNICODE) ?>;
    const preselDiseases = <?= json_encode(parse_compare_entities($diseaseInput), JSON_UNESCAPED_UNICODE) ?>;

    const state = {
        drug: { all: [], selected: new Set(), el: { tags: 'drug-tags', list: 'drug-list', search: 'drug-search', hidden: 'drugs-hidden' } },
        disease: { all: [], selected: new Set(), el: { tags: 'disease-tags', list: 'disease-list', search: 'disease-search', hidden: 'diseases-hidden' } }
    };

    function esc(t) { const d = document.createElement('span'); d.textContent = t; return d.innerHTML; }

    function render(type) {
        const s = state[type];
        const q = document.getElementById(s.el.search).value.toLowerCase();
        const listEl = document.getElementById(s.el.list);
        const filtered = s.all.filter(i => !q || `${i.id} ${i.name}`.toLowerCase().includes(q));

        if (!s.all.length) { listEl.innerHTML = '<div class="entity-picker-msg">Khong co du lieu</div>'; renderTags(type); return; }
        if (!filtered.length) { listEl.innerHTML = '<div class="entity-picker-msg">Khong tim thay</div>'; renderTags(type); return; }

        listEl.innerHTML = filtered.map(i => {
            const sel = s.selected.has(i.id);
            const dis = !sel && s.selected.size >= MAX;
            return `<div class="entity-picker-item${sel ? ' ep-selected' : ''}${dis ? ' ep-disabled' : ''}" data-id="${esc(i.id)}">` +
                `<input type="checkbox"${sel ? ' checked' : ''}${dis ? ' disabled' : ''}>` +
                `<span class="entity-picker-item-name">${esc(i.name)}</span>` +
                `<span class="entity-picker-id">${esc(i.id)}</span></div>`;
        }).join('') + `<div class="entity-picker-count">Da chon ${s.selected.size}/${MAX}</div>`;

        listEl.querySelectorAll('.entity-picker-item').forEach(el => {
            el.addEventListener('click', () => {
                const id = el.dataset.id;
                if (s.selected.has(id)) { s.selected.delete(id); }
                else if (s.selected.size < MAX) { s.selected.add(id); }
                else { return; }
                sync(type); render(type);
            });
        });
        renderTags(type);
    }

    function renderTags(type) {
        const s = state[type];
        const cls = type === 'drug' ? 'entity-tag-drug' : 'entity-tag-disease';
        const el = document.getElementById(s.el.tags);
        el.innerHTML = Array.from(s.selected).map(id => {
            const item = s.all.find(i => i.id === id);
            return `<span class="entity-tag ${cls}" data-id="${esc(id)}">${esc(item ? item.name : id)} <span class="entity-tag-remove">×</span></span>`;
        }).join('');
        el.querySelectorAll('.entity-tag').forEach(tag => {
            tag.addEventListener('click', () => { s.selected.delete(tag.dataset.id); sync(type); render(type); });
        });
    }

    function sync(type) {
        document.getElementById(state[type].el.hidden).value = Array.from(state[type].selected).join(',');
    }

    async function load(dataset) {
        for (const type of ['drug', 'disease']) {
            document.getElementById(state[type].el.list).innerHTML = '<div class="entity-picker-msg">Dang tai...</div>';
            document.getElementById(state[type].el.search).value = '';
        }
        try {
            const res = await fetch(`${API}/entities?dataset=${encodeURIComponent(dataset)}`);
            if (!res.ok) throw new Error();
            const data = await res.json();
            state.drug.all = data.drugs || [];
            state.disease.all = data.diseases || [];
        } catch { state.drug.all = []; state.disease.all = []; }
        render('drug'); render('disease');
    }

    function preselect(type, values) {
        const s = state[type];
        values.forEach(v => {
            const vl = v.toLowerCase();
            const m = s.all.find(i => i.id.toLowerCase() === vl || i.name.toLowerCase() === vl);
            if (m && s.selected.size < MAX) s.selected.add(m.id);
        });
        sync(type);
    }

    const sel = document.getElementById('dataset-select');
    sel.addEventListener('change', () => { state.drug.selected.clear(); state.disease.selected.clear(); load(sel.value); });
    document.getElementById('drug-search').addEventListener('input', () => render('drug'));
    document.getElementById('disease-search').addEventListener('input', () => render('disease'));

    load(sel.value).then(() => {
        preselect('drug', preselDrugs);
        preselect('disease', preselDiseases);
        render('drug'); render('disease');
    });
})();
</script>
</body>
</html>
