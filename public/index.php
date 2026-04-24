<?php
require_once __DIR__ . '/../app/services/PredictionService.php';
require_login();

$user = current_user();
$resultData = null;
$error = null;
$queryType = $_POST['query_type'] ?? 'drug_to_disease';
$inputText = trim($_POST['input_text'] ?? '');
$dataset = $_POST['dataset'] ?? 'C-dataset';
$topK = max(1, min(20, (int) ($_POST['top_k'] ?? 10)));
$apiHealthy = PredictionService::isApiHealthy();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if ($inputText === '') {
        $error = 'Vui long nhap ten thuoc hoac benh.';
    } elseif (!$apiHealthy) {
        $error = 'He thong AI hien dang ngoai tuyen. Vui long kiem tra lai server.';
    } else {
        try {
            $resultData = PredictionService::callPythonApi($queryType, $inputText, $topK, $dataset);
            PredictionService::saveHistory((int) $user['id'], $queryType, $inputText, $topK, $resultData);
            flash('success', 'Du doan thanh cong');
            $_SESSION['latest_graph'] = $resultData['graph'] ?? ['nodes' => [], 'links' => []];
        } catch (Throwable $e) {
            $error = $e->getMessage();
        }
    }
}

$success = flash('success');
$graph = $resultData['graph'] ?? ($_SESSION['latest_graph'] ?? ['nodes' => [], 'links' => []]);
$graphStats = ['drug' => 0, 'disease' => 0, 'protein' => 0];
foreach (($graph['nodes'] ?? []) as $node) {
    $type = strtolower((string) ($node['type'] ?? ''));
    if (array_key_exists($type, $graphStats)) {
        $graphStats[$type]++;
    }
}

$historyStmt = db()->prepare('SELECT * FROM prediction_requests WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 8');
$historyStmt->execute(['user_id' => $user['id']]);
$histories = $historyStmt->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HGT AI Dashboard</title>
    <link rel="stylesheet" href="assets/style.css">
    <script src="https://unpkg.com/force-graph"></script>
</head>
<body>
<div id="loader" class="loading-overlay">Dang phan tich do thi...</div>
<div id="toast-container" class="toast-container"></div>

<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">AMNTDDA Dashboard</div>
            <div class="muted">He thong du doan lien ket thuoc - benh voi giao dien thong nhat, sang va de theo doi.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="compare_models.php">So sanh 2 mo hinh</a>
            <a class="btn btn-ghost" href="history.php">Lich su</a>
            <?php if (($user['role'] ?? '') === 'admin'): ?>
                <a class="btn btn-ghost" href="admin.php">Quan tri</a>
            <?php endif; ?>
            <a class="btn btn-danger" href="logout.php">Dang xuat</a>
        </div>
    </div>

    <div class="glass-card hero-banner">
        <div class="hero-grid">
            <div>
                <div class="badge badge-drug spacer-md">AI Graph Prediction</div>
                <h1>Tra cuu lien ket thuoc - benh nhanh hon va de doc hon</h1>
                <p class="muted spacer-lg">Nhap ten thuoc, benh hoac ma ID. He thong se chay mo hinh HGT va tra ve top-k lien ket co xac suat cao nhat, kem do thi 3D truc quan.</p>
            </div>
            <div class="status-card">
                <div class="label">Trang thai AI API</div>
                <div class="spacer-md"></div>
                <span class="badge <?= $apiHealthy ? 'badge-success' : 'badge-neutral' ?>">
                    <?= $apiHealthy ? 'Online' : 'Offline' ?>
                </span>
                <p class="muted spacer-md"></p>
                <p class="muted"><?= $apiHealthy ? 'Server AI dang san sang xu ly du doan.' : 'Can khoi dong Python API o cong 8000.' ?></p>
            </div>
        </div>
    </div>

    <?php if (!$apiHealthy): ?>
        <div class="alert alert-error">Canh bao: Python HGT-API o cong 8000 dang ngat ket noi.</div>
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
                <h2>He thong du doan lien ket Thuoc - Benh</h2>
                <p class="muted">Chon dataset, nhap tu khoa va lay ket qua top-k theo nhu cau.</p>
            </div>
        </div>

        <form method="post" onsubmit="document.getElementById('loader').style.display='grid'">
            <div class="search-grid">
                <div class="form-group">
                    <label class="label">Bo du lieu</label>
                    <select class="select" name="dataset">
                        <option value="C-dataset" <?= $dataset === 'C-dataset' ? 'selected' : '' ?>>Dataset C</option>
                        <option value="B-dataset" <?= $dataset === 'B-dataset' ? 'selected' : '' ?>>Dataset B</option>
                        <option value="F-dataset" <?= $dataset === 'F-dataset' ? 'selected' : '' ?>>Dataset F</option>
                    </select>
                </div>

                <div class="form-group">
                    <label class="label">Tu khoa</label>
                    <input class="input" type="text" name="input_text" value="<?= e($inputText) ?>" placeholder="Goserelin, DB00014, D102100..." required>
                </div>

                <div class="form-group">
                    <label class="label label-center">Loai truy van</label>
                    <div class="query-toggle-container">
                        <input type="checkbox" id="query_toggle" class="toggle-input" name="query_type_toggle" <?= $queryType === 'disease_to_drug' ? 'checked' : '' ?> onchange="this.nextElementSibling.nextElementSibling.value = this.checked ? 'disease_to_drug' : 'drug_to_disease'">
                        <label for="query_toggle" class="toggle-label" title="Thuoc / Benh">
                            <div class="toggle-slider"></div>
                            <div class="toggle-text"><span class="text-top">Thuoc</span><span class="text-bottom">Benh</span></div>
                        </label>
                        <input type="hidden" name="query_type" value="<?= e($queryType) ?>">
                    </div>
                </div>

                <div class="form-group">
                    <label class="label">Lay Top-K</label>
                    <div class="toolbar-inline">
                        <input class="input input-top-k" type="number" name="top_k" min="1" max="20" value="<?= e((string) $topK) ?>">
                        <button class="btn" type="submit" <?= !$apiHealthy ? 'disabled' : '' ?>>Du doan</button>
                    </div>
                </div>
            </div>
        </form>
    </div>

    <div class="grid grid-2">
        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Ket qua phan tich HGT</h3>
                    <p class="muted">Danh sach ket qua co xac suat du doan cao nhat.</p>
                </div>
                <div class="badge badge-drug"><?= $topK ?> results</div>
            </div>

            <?php if ($resultData): ?>
                <div class="table-container">
                    <table class="table">
                        <thead>
                        <tr>
                            <th>Hang</th>
                            <th>Thuc the tiem nang</th>
                            <th>Loai</th>
                            <th>Xac suat</th>
                        </tr>
                        </thead>
                        <tbody>
                        <?php foreach (($resultData['results'] ?? []) as $index => $item): ?>
                            <tr>
                                <td class="muted">#<?= $index + 1 ?></td>
                                <td>
                                    <div class="result-title"><?= e((string) $item['name']) ?></div>
                                    <div class="muted mono">ID: <?= e((string) $item['id']) ?></div>
                                </td>
                                <td><span class="badge <?= ($item['type'] ?? '') === 'drug' ? 'badge-drug' : 'badge-disease' ?>"><?= e((string) $item['type']) ?></span></td>
                                <td class="score-text"><?= e(number_format((float) $item['score'], 4)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            <?php else: ?>
                <div class="center-empty">
                    <div class="empty-icon">+</div>
                    <p>Nhap tu khoa va nhan Du doan de bat dau phan tich.</p>
                </div>
            <?php endif; ?>
        </div>

        <div class="glass-card">
            <div class="section-header">
                <div>
                    <h3>Lich su gan day</h3>
                    <p class="muted">8 luot tra cuu gan nhat cua tai khoan hien tai.</p>
                </div>
            </div>

            <div class="table-container">
                <table class="table">
                    <thead>
                    <tr>
                        <th>Input</th>
                        <th class="align-right">Ket qua</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php if (empty($histories)): ?>
                        <tr><td colspan="2" class="center-empty">Chua co du lieu</td></tr>
                    <?php endif; ?>
                    <?php foreach ($histories as $item): ?>
                        <tr>
                            <td>
                                <span class="muted"><?= e((string) $item['query_type']) ?></span><br>
                                <strong><?= e((string) $item['input_text']) ?></strong>
                            </td>
                            <td class="align-right"><span class="badge badge-neutral">Top-<?= e((string) $item['top_k']) ?></span></td>
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
                <h3>So do mang lien ket truc quan</h3>
                <p class="muted">Do thi duoc trinh bay theo 3 nhom mau ro rang: thuoc, benh va protein cau noi.</p>
            </div>
            <div class="graph-legend">
                <div class="legend-item">
                    <span class="legend-dot legend-dot-drug"></span>
                    <span>Thuoc</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot legend-dot-disease"></span>
                    <span>Benh</span>
                </div>
                <div class="legend-item">
                    <span class="legend-dot legend-dot-protein"></span>
                    <span>Protein</span>
                </div>
            </div>
        </div>

        <div class="graph-insights">
            <div class="graph-insight">
                <span class="legend-dot legend-dot-drug"></span>
                <div>
                    <strong><?= $graphStats['drug'] ?> nut thuoc</strong>
                    <p>Thuoc duoc ve theo kieu khoi cau lien ket, giu mau xanh duong de nhin ra nhanh vai tro cua nut trung tam hoac ung vien.</p>
                </div>
            </div>
            <div class="graph-insight">
                <span class="legend-dot legend-dot-disease"></span>
                <div>
                    <strong><?= $graphStats['disease'] ?> nut benh</strong>
                    <p>Benh dung mau hong do va duoc dat o lop ngoai, giup so do giong cach trinh bay phan tu nhung van de tach nhom.</p>
                </div>
            </div>
            <div class="graph-insight">
                <span class="legend-dot legend-dot-protein"></span>
                <div>
                    <strong><?= $graphStats['protein'] ?> nut protein</strong>
                    <p>Protein la lop lien ket o giua, dong vai tro nhu cau noi trong so do dang ball-and-stick de nhin luong lien quan ro hon.</p>
                </div>
            </div>
        </div>

        <?php if (empty($graph['nodes'])): ?>
            <div class="empty-panel">Chua co du lieu do thi. Vui long thuc hien du doan.</div>
        <?php else: ?>
            <div class="graph-note">
                So do duoc chuan hoa theo phong cach lien ket hoa hoc: nut truy van goc nam o tam, protein tao vong lien ket trung gian va cac ket qua nam o lop ngoai.
            </div>
            <div id="graph3d"></div>
        <?php endif; ?>
    </div>
</div>

<script>
const graphData = <?= json_encode($graph, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const apiNote = <?= json_encode($resultData['note'] ?? '', JSON_UNESCAPED_UNICODE) ?>;
const phpSuccess = <?= json_encode($success ?? '', JSON_UNESCAPED_UNICODE) ?>;

function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    const isWarn = message.toLowerCase().includes('canh bao');
    const isSuccess = message.toLowerCase().includes('thanh cong');
    toast.className = `toast ${isWarn ? 'toast-warn' : ''} ${isSuccess ? 'toast-success' : ''}`;
    toast.innerHTML = `<div class="toast-icon">${isWarn ? '!' : (isSuccess ? '+' : 'i')}</div><div class="toast-message">${message}</div>`;
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 100);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 5000);
}

if (apiNote) showToast(apiNote);
if (phpSuccess) showToast(phpSuccess);

const container = document.getElementById('graph3d');
if (container && graphData && Array.isArray(graphData.nodes) && graphData.nodes.length > 0) {
    const TYPE_CONFIG = {
        drug: { label: 'Thuoc', color: '#2563eb', glow: 'rgba(37, 99, 235, 0.18)', size: 16 },
        disease: { label: 'Benh', color: '#e85d75', glow: 'rgba(232, 93, 117, 0.18)', size: 16 },
        protein: { label: 'Protein', color: '#14b8a6', glow: 'rgba(20, 184, 166, 0.18)', size: 13 },
        default: { label: 'Node', color: '#64748b', glow: 'rgba(100, 116, 139, 0.16)', size: 12 }
    };

    const normalizeType = (type) => {
        const value = String(type || '').toLowerCase();
        return TYPE_CONFIG[value] ? value : 'default';
    };

    const truncateLabel = (value, limit = 24) => {
        if (!value) return 'Unknown';
        return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
    };

    const buildLayoutGraph = (rawGraph) => {
        const nodes = (rawGraph.nodes || []).map((node, index) => ({
            ...node,
            type: normalizeType(node.type),
            label: node.label || node.name || node.actual_id || node.id,
            isSource: index === 0
        }));
        const links = (rawGraph.links || []).map((link) => ({ ...link }));

        const sourceType = nodes[0]?.type || 'drug';
        const targetNode = nodes.find((node, index) => index > 0 && node.type !== 'protein' && node.type !== sourceType);
        const targetType = targetNode?.type || (sourceType === 'drug' ? 'disease' : 'drug');
        const columns = { [sourceType]: -320, protein: 0, [targetType]: 320 };

        ['drug', 'protein', 'disease', 'default'].forEach((type, fallbackIndex) => {
            if (!Object.prototype.hasOwnProperty.call(columns, type)) {
                columns[type] = (-320 + fallbackIndex * 210);
            }
        });

        const groupedNodes = { drug: [], disease: [], protein: [], default: [] };
        nodes.forEach((node) => {
            groupedNodes[node.type] = groupedNodes[node.type] || [];
            groupedNodes[node.type].push(node);
        });

        const applyColumn = (list, type) => {
            const total = list.length;
            if (!total) {
                return;
            }
            const spacing = total === 1 ? 0 : Math.min(140, 520 / Math.max(total - 1, 1));
            const start = -((total - 1) * spacing) / 2;

            list.forEach((node, index) => {
                const config = TYPE_CONFIG[node.type] || TYPE_CONFIG.default;
                node.fx = columns[type] ?? 0;
                node.fy = start + index * spacing;
                node.val = config.size + (node.isSource ? 6 : 0);
            });
        };

        applyColumn(groupedNodes[sourceType] || [], sourceType);
        applyColumn(groupedNodes.protein || [], 'protein');
        applyColumn(groupedNodes[targetType] || [], targetType);

        Object.entries(groupedNodes).forEach(([type, list]) => {
            if (type !== sourceType && type !== 'protein' && type !== targetType) {
                applyColumn(list, type);
            }
        });

        return { nodes, links };
    };

    const truncateNodeLabel = (value, limit = 24) => {
        if (!value) return 'Unknown';
        return value.length > limit ? `${value.slice(0, limit - 3)}...` : value;
    };

    const drawRoundedRect = (ctx, x, y, width, height, radius) => {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    };

    const preparedGraph = buildLayoutGraph(graphData);
    const Graph = ForceGraph()(container)
        .graphData(preparedGraph)
        .nodeId('id')
        .backgroundColor('rgba(255,255,255,0)')
        .nodeLabel((node) => `${node.label} (${TYPE_CONFIG[node.type]?.label || 'Node'})`)
        .linkColor((link) => {
            const sourceType = normalizeType(link.source?.type || link.source);
            const targetType = normalizeType(link.target?.type || link.target);
            if (sourceType === 'protein' || targetType === 'protein') {
                return 'rgba(20, 184, 166, 0.28)';
            }
            return 'rgba(100, 116, 139, 0.26)';
        })
        .linkWidth((link) => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return sourceId === preparedGraph.nodes[0]?.id || targetId === preparedGraph.nodes[0]?.id ? 2.4 : 1.6;
        })
        .linkDirectionalParticles(2)
        .linkDirectionalParticleWidth(1.8)
        .linkDirectionalParticleColor((link) => {
            const sourceType = normalizeType(link.source?.type || link.source);
            return TYPE_CONFIG[sourceType]?.color || TYPE_CONFIG.default.color;
        })
        .enableNodeDrag(false)
        .cooldownTicks(0)
        .nodeCanvasObject((node, ctx, globalScale) => {
            const config = TYPE_CONFIG[node.type] || TYPE_CONFIG.default;
            const radius = node.val || config.size;
            const fontSize = Math.max(12 / globalScale, 11);
            const label = truncateNodeLabel(node.label, node.isSource ? 28 : 22);
            ctx.font = `700 ${fontSize}px "Plus Jakarta Sans", sans-serif`;
            const pillWidth = ctx.measureText(label).width + fontSize * 1.6;
            const pillHeight = fontSize * 1.8;
            const pillX = node.x - (pillWidth / 2);
            const pillY = node.y + radius + 10;

            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 8, 0, 2 * Math.PI, false);
            ctx.fillStyle = config.glow;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
            ctx.fillStyle = config.color;
            ctx.fill();
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();

            drawRoundedRect(ctx, pillX, pillY, pillWidth, pillHeight, pillHeight / 2);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.96)';
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(126, 108, 79, 0.14)';
            ctx.stroke();

            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#1d2433';
            ctx.fillText(label, node.x, pillY + (pillHeight / 2));

            if (node.isSource) {
                ctx.font = `800 ${Math.max(10 / globalScale, 9)}px "Plus Jakarta Sans", sans-serif`;
                ctx.fillStyle = config.color;
                ctx.fillText('TRUY VAN GOC', node.x, node.y - radius - 14);
            }
        });

    Graph.zoom(1.08, 0);
    Graph.centerAt(0, 0, 0);

    const clampChannel = (value) => Math.max(0, Math.min(255, value));
    const shadeHex = (hex, amount) => {
        const safeHex = String(hex || '#64748b').replace('#', '');
        const num = Number.parseInt(safeHex, 16);
        const red = clampChannel(((num >> 16) & 255) + amount);
        const green = clampChannel(((num >> 8) & 255) + amount);
        const blue = clampChannel((num & 255) + amount);
        return `rgb(${red}, ${green}, ${blue})`;
    };

    const placeRingNodes = (nodes, radius, startAngle = -Math.PI / 2) => {
        if (!nodes.length) {
            return;
        }
        const step = (Math.PI * 2) / Math.max(nodes.length, 1);
        nodes.forEach((node, index) => {
            const angle = startAngle + (index * step);
            node.fx = Math.cos(angle) * radius;
            node.fy = Math.sin(angle) * radius;
        });
    };

    const buildChemicalLayoutGraph = (rawGraph) => {
        const nodes = (rawGraph.nodes || []).map((node, index) => ({
            ...node,
            type: normalizeType(node.type),
            label: node.label || node.name || node.actual_id || node.id,
            isSource: index === 0
        }));
        const links = (rawGraph.links || []).map((link) => ({ ...link }));
        const sourceNode = nodes[0] || null;
        const proteinNodes = nodes.filter((node) => !node.isSource && node.type === 'protein');
        const outerNodes = nodes.filter((node) => !node.isSource && node.type !== 'protein');

        if (sourceNode) {
            sourceNode.fx = 0;
            sourceNode.fy = 0;
            sourceNode.val = 48;
        }

        placeRingNodes(proteinNodes, proteinNodes.length <= 3 ? 118 : 142);
        placeRingNodes(outerNodes, outerNodes.length <= 4 ? 238 : 282, -Math.PI / 3);

        proteinNodes.forEach((node) => { node.val = 28; });
        outerNodes.forEach((node) => { node.val = 38; });

        return { nodes, links };
    };

    container.innerHTML = '';
    const chemicalGraphData = buildChemicalLayoutGraph(graphData);
    const ChemicalGraph = ForceGraph()(container)
        .graphData(chemicalGraphData)
        .nodeId('id')
        .backgroundColor('rgba(255,255,255,0)')
        .nodeLabel((node) => `${node.label} (${TYPE_CONFIG[node.type]?.label || 'Node'})`)
        .linkColor((link) => {
            const sourceType = normalizeType(link.source?.type || link.source);
            const targetType = normalizeType(link.target?.type || link.target);
            if (sourceType === 'protein' || targetType === 'protein') {
                return 'rgba(20, 184, 166, 0.55)';
            }
            return 'rgba(126, 108, 79, 0.42)';
        })
        .linkWidth((link) => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return sourceId === chemicalGraphData.nodes[0]?.id || targetId === chemicalGraphData.nodes[0]?.id ? 5.2 : 3.4;
        })
        .enableNodeDrag(false)
        .cooldownTicks(0)
        .nodeCanvasObject((node, ctx, globalScale) => {
            const config = TYPE_CONFIG[node.type] || TYPE_CONFIG.default;
            const radius = node.val || 30;
            const label = truncateNodeLabel(node.label, node.isSource ? 28 : 18);
            const titleFontSize = Math.max((node.isSource ? 18 : 14) / globalScale, 11);
            const subFontSize = Math.max(10 / globalScale, 8);
            const gradient = ctx.createRadialGradient(
                node.x - radius * 0.42,
                node.y - radius * 0.48,
                radius * 0.15,
                node.x,
                node.y,
                radius * 1.05
            );

            gradient.addColorStop(0, shadeHex(config.color, 95));
            gradient.addColorStop(0.35, shadeHex(config.color, 25));
            gradient.addColorStop(1, shadeHex(config.color, -42));

            ctx.save();
            ctx.shadowColor = config.glow;
            ctx.shadowBlur = radius * 1.1;
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
            ctx.fillStyle = gradient;
            ctx.fill();
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.88)';
            ctx.stroke();
            ctx.restore();

            ctx.beginPath();
            ctx.arc(node.x - radius * 0.32, node.y - radius * 0.36, radius * 0.24, 0, 2 * Math.PI, false);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.34)';
            ctx.fill();

            ctx.font = `800 ${titleFontSize}px "Plus Jakarta Sans", sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#141b28';
            ctx.fillText(label, node.x, node.y - radius - (node.isSource ? 26 : 18));

            ctx.font = `700 ${subFontSize}px "Plus Jakarta Sans", sans-serif`;
            ctx.fillStyle = node.isSource ? config.color : '#6b7280';
            ctx.fillText(node.isSource ? 'TRUY VAN GOC' : (TYPE_CONFIG[node.type]?.label || 'Node'), node.x, node.y + radius + 14);
        });

    ChemicalGraph.zoom(1.02, 0);
    ChemicalGraph.centerAt(0, 0, 0);
}
</script>
<script>
(() => {
    const host = document.getElementById('graph3d');
    if (!host || !graphData || !Array.isArray(graphData.nodes) || graphData.nodes.length === 0) {
        return;
    }

    const TYPE_CONFIG = {
        drug: { label: 'Thuoc', color: '#2563eb', glow: 'rgba(37, 99, 235, 0.20)', size: 38 },
        disease: { label: 'Benh', color: '#e85d75', glow: 'rgba(232, 93, 117, 0.20)', size: 38 },
        protein: { label: 'Protein', color: '#14b8a6', glow: 'rgba(20, 184, 166, 0.18)', size: 28 },
        default: { label: 'Node', color: '#64748b', glow: 'rgba(100, 116, 139, 0.16)', size: 24 }
    };

    const normalizeType = (type) => {
        const value = String(type || '').toLowerCase();
        return TYPE_CONFIG[value] ? value : 'default';
    };

    const truncateLabel = (value, limit = 24) => {
        if (!value) return 'Unknown';
        return value.length > limit ? `${value.slice(0, Math.max(limit - 3, 1))}...` : value;
    };

    const clampChannel = (value) => Math.max(0, Math.min(255, value));
    const shadeHex = (hex, amount) => {
        const safeHex = String(hex || '#64748b').replace('#', '');
        const num = Number.parseInt(safeHex, 16);
        const red = clampChannel(((num >> 16) & 255) + amount);
        const green = clampChannel(((num >> 8) & 255) + amount);
        const blue = clampChannel((num & 255) + amount);
        return `rgb(${red}, ${green}, ${blue})`;
    };

    const placeRingNodes = (nodes, radius, startAngle = -Math.PI / 2) => {
        if (!nodes.length) {
            return;
        }
        const step = (Math.PI * 2) / Math.max(nodes.length, 1);
        nodes.forEach((node, index) => {
            const angle = startAngle + (index * step);
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            node.x = x;
            node.y = y;
            node.fx = x;
            node.fy = y;
        });
    };

    const layoutGraph = (() => {
        const nodes = (graphData.nodes || []).map((node, index) => ({
            ...node,
            type: normalizeType(node.type),
            label: node.label || node.name || node.actual_id || node.id,
            isSource: index === 0
        }));
        const links = (graphData.links || []).map((link) => ({ ...link }));
        const sourceNode = nodes[0] || null;
        const proteinNodes = nodes.filter((node) => !node.isSource && node.type === 'protein');
        const outerNodes = nodes.filter((node) => !node.isSource && node.type !== 'protein');

        if (sourceNode) {
            sourceNode.x = 0;
            sourceNode.y = 0;
            sourceNode.fx = 0;
            sourceNode.fy = 0;
            sourceNode.val = 52;
        }

        placeRingNodes(proteinNodes, proteinNodes.length <= 3 ? 120 : 148);
        placeRingNodes(outerNodes, outerNodes.length <= 4 ? 235 : 285, -Math.PI / 3);

        proteinNodes.forEach((node) => { node.val = TYPE_CONFIG.protein.size; });
        outerNodes.forEach((node) => {
            const config = TYPE_CONFIG[node.type] || TYPE_CONFIG.default;
            node.val = config.size;
        });

        return { nodes, links };
    })();

    host.innerHTML = '';

    const graph = ForceGraph()(host)
        .graphData(layoutGraph)
        .nodeId('id')
        .backgroundColor('rgba(255,255,255,0)')
        .nodeLabel((node) => `${node.label} (${TYPE_CONFIG[node.type]?.label || 'Node'})`)
        .linkColor((link) => {
            const sourceType = normalizeType(link.source?.type || link.source);
            const targetType = normalizeType(link.target?.type || link.target);
            if (sourceType === 'protein' || targetType === 'protein') {
                return 'rgba(20, 184, 166, 0.55)';
            }
            return 'rgba(126, 108, 79, 0.42)';
        })
        .linkWidth((link) => {
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            return sourceId === layoutGraph.nodes[0]?.id || targetId === layoutGraph.nodes[0]?.id ? 5.2 : 3.4;
        })
        .enableNodeDrag(false)
        .warmupTicks(1)
        .cooldownTicks(1)
        .nodeCanvasObject((node, ctx, globalScale) => {
            const config = TYPE_CONFIG[node.type] || TYPE_CONFIG.default;
            const radius = node.val || config.size;
            const label = truncateLabel(node.label, node.isSource ? 28 : 18);
            const titleFontSize = Math.max((node.isSource ? 18 : 14) / globalScale, 11);
            const subFontSize = Math.max(10 / globalScale, 8);
            const gradient = ctx.createRadialGradient(
                node.x - radius * 0.42,
                node.y - radius * 0.48,
                radius * 0.15,
                node.x,
                node.y,
                radius * 1.05
            );

            gradient.addColorStop(0, shadeHex(config.color, 95));
            gradient.addColorStop(0.35, shadeHex(config.color, 25));
            gradient.addColorStop(1, shadeHex(config.color, -42));

            ctx.save();
            ctx.shadowColor = config.glow;
            ctx.shadowBlur = radius * 1.1;
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
            ctx.fillStyle = gradient;
            ctx.fill();
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.88)';
            ctx.stroke();
            ctx.restore();

            ctx.beginPath();
            ctx.arc(node.x - radius * 0.32, node.y - radius * 0.36, radius * 0.24, 0, 2 * Math.PI, false);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.34)';
            ctx.fill();

            ctx.font = `800 ${titleFontSize}px "Plus Jakarta Sans", sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#141b28';
            ctx.fillText(label, node.x, node.y - radius - (node.isSource ? 26 : 18));

            ctx.font = `700 ${subFontSize}px "Plus Jakarta Sans", sans-serif`;
            ctx.fillStyle = node.isSource ? config.color : '#6b7280';
            ctx.fillText(node.isSource ? 'TRUY VAN GOC' : (TYPE_CONFIG[node.type]?.label || 'Node'), node.x, node.y + radius + 14);
        })
        .onEngineStop(() => {
            graph.centerAt(0, 0, 250);
            graph.zoom(1.02, 250);
        });

    graph.centerAt(0, 0, 0);
})();
</script>
</body>
</html>
