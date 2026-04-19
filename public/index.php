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
        $error = 'Vui lòng nhập tên thuốc hoặc bệnh.';
    } elseif (!$apiHealthy) {
        $error = 'Hệ thống AI hiện đang ngoại tuyến. Vui lòng kiểm tra lại server.';
    } else {
        try {
            $resultData = PredictionService::callPythonApi($queryType, $inputText, $topK, $dataset);
            PredictionService::saveHistory((int) $user['id'], $queryType, $inputText, $topK, $resultData);
            flash('success', 'Thành công');
            $_SESSION['latest_graph'] = $resultData['graph'] ?? ['nodes' => [], 'links' => []];
        } catch (Throwable $e) {
            $error = $e->getMessage();
        }
    }
}

$success = flash('success');
$graph = $resultData['graph'] ?? ($_SESSION['latest_graph'] ?? ['nodes' => [], 'links' => []]);

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
    <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
<div id="loader" class="loading-overlay" style="display:none; position:fixed; inset:0; background:rgba(2,6,23,0.78); backdrop-filter:blur(10px); z-index:9999; place-items:center; color:#fff; font-weight:700;">Đang phân tích đồ thị...</div>
<div id="toast-container" class="toast-container"></div>

<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">AMNTDDA <span style="font-weight:600; font-size:.72em; opacity:.7; -webkit-text-fill-color:#cbd5e1;">HGT Link Predictor</span></div>
            <div class="muted">Dự đoán liên kết thuốc - bệnh với giao diện hiện đại</div>
        </div>
        <div class="nav-links">
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="history.php">Lịch sử</a>
            <?php if (($user['role'] ?? '') === 'admin'): ?>
                <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="admin.php">Quản trị</a>
            <?php endif; ?>
            <a class="btn" style="background: rgba(244, 63, 94, 0.16); box-shadow:none;" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="glass-card" style="margin-bottom: 24px; padding: 28px; background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(15,23,42,0.72));">
        <div class="grid" style="grid-template-columns: 1.6fr .9fr; align-items:center;">
            <div>
                <div class="badge" style="background: rgba(96,165,250,0.14); color:#bfdbfe; border:1px solid rgba(96,165,250,0.2); margin-bottom:14px;">AI Graph Prediction</div>
                <h1 style="font-size: 2.2rem; margin-bottom: 10px;">Tìm liên kết thuốc - bệnh nhanh hơn, trực quan hơn</h1>
                <p class="muted" style="max-width: 760px;">Nhập tên thuốc hoặc bệnh, hệ thống sẽ chạy mô hình HGT và trả về top-k mối liên hệ có xác suất cao nhất.</p>
            </div>
            <div class="glass-card" style="padding:18px; background: rgba(15,23,42,0.72);">
                <div class="label">Trạng thái API</div>
                <div style="display:flex; align-items:center; gap:10px; margin-top:10px;">
                    <span class="badge" style="background: <?= $apiHealthy ? 'rgba(34,197,94,0.14)' : 'rgba(244,63,94,0.14)' ?>; color: <?= $apiHealthy ? '#86efac' : '#fda4af' ?>; border-color: <?= $apiHealthy ? 'rgba(34,197,94,0.24)' : 'rgba(244,63,94,0.24)' ?>;">
                        <?= $apiHealthy ? 'ONLINE' : 'OFFLINE' ?>
                    </span>
                    <span class="muted"><?= $apiHealthy ? 'Server AI đang sẵn sàng.' : 'Hệ thống AI đang ngắt kết nối.' ?></span>
                </div>
            </div>
        </div>
    </div>

    <?php if (!$apiHealthy): ?><div class="alert alert-error">Cảnh báo: Python HGT-API (Port 8000) đang ngắt kết nối.</div><?php endif; ?>
    <?php if ($error): ?><div class="alert alert-error"><?= e($error) ?></div><?php endif; ?>
    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>

    <div class="glass-card search-container" style="margin-bottom: 24px;">
        <h2 style="margin-bottom: 18px; font-size: 1.25rem;">Hệ thống dự đoán liên kết Thuốc - Bệnh</h2>
        <form method="post" onsubmit="document.getElementById('loader').style.display='grid'">
            <div class="search-grid">
                <div class="form-group">
                    <label class="label">Bộ dữ liệu</label>
                    <select class="select" name="dataset">
                        <option value="C-dataset" <?= $dataset === 'C-dataset' ? 'selected' : '' ?>>Dataset C (Chuẩn)</option>
                        <option value="B-dataset" <?= $dataset === 'B-dataset' ? 'selected' : '' ?>>Dataset B</option>
                        <option value="F-dataset" <?= $dataset === 'F-dataset' ? 'selected' : '' ?>>Dataset F</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="label">Từ khóa (Tên hoặc ID)</label>
                    <input class="input" type="text" name="input_text" value="<?= e($inputText) ?>" placeholder="Goserelin, DB00014, D102100..." required>
                </div>
                <div class="form-group">
                    <label class="label" style="text-align:center;">Thực thể tìm</label>
                    <div class="query-toggle-container">
                        <input type="checkbox" id="query_toggle" class="toggle-input" name="query_type_toggle" <?= $queryType === 'disease_to_drug' ? 'checked' : '' ?> onchange="this.nextElementSibling.nextElementSibling.value = this.checked ? 'disease_to_drug' : 'drug_to_disease'">
                        <label for="query_toggle" class="toggle-label" title="Thuốc / Bệnh">
                            <div class="toggle-slider"></div>
                            <div class="toggle-text"><span class="text-top">Thuốc</span><span class="text-bottom">Bệnh</span></div>
                        </label>
                        <input type="hidden" name="query_type" value="<?= e($queryType) ?>">
                    </div>
                </div>
                <div class="form-group">
                    <label class="label">Lấy Top-K</label>
                    <div style="display:flex; gap:10px;">
                        <input class="input input-top-k" type="number" name="top_k" min="1" max="20" value="<?= e((string) $topK) ?>">
                        <button class="btn" type="submit" <?= !$apiHealthy ? 'disabled' : '' ?>>Dự đoán</button>
                    </div>
                </div>
            </div>
        </form>
    </div>

    <div class="grid grid-2">
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:end; gap:12px; margin-bottom: 14px; flex-wrap:wrap;">
                <div>
                    <h3>Kết quả phân tích HGT</h3>
                    <p class="muted">Danh sách top-k theo xác suất dự đoán.</p>
                </div>
                <div class="badge badge-drug"><?= $topK ?> results</div>
            </div>
            <?php if ($resultData): ?>
                <div class="table-container">
                    <table class="table">
                        <thead><tr><th>Hạng</th><th>Thực thể tiềm năng</th><th>Loại</th><th>Xác suất</th></tr></thead>
                        <tbody>
                        <?php foreach (($resultData['results'] ?? []) as $index => $item): ?>
                            <tr>
                                <td style="font-weight:700; color:#94a3b8;">#<?= $index + 1 ?></td>
                                <td>
                                    <div style="font-weight:700;"><?= e((string) $item['name']) ?></div>
                                    <div style="font-size:.78rem; color:#94a3b8; font-family:'JetBrains Mono', monospace;">ID: <?= e((string) $item['id']) ?></div>
                                </td>
                                <td><span class="badge <?= ($item['type'] ?? '') === 'drug' ? 'badge-drug' : 'badge-disease' ?>"><?= e((string) $item['type']) ?></span></td>
                                <td class="score-text"><?= e(number_format((float) $item['score'], 4)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            <?php else: ?>
                <div style="padding: 64px 0; text-align:center; color: var(--text-muted);">
                    <div style="font-size: 40px; margin-bottom: 10px;">◌</div>
                    <p>Nhập từ khóa và nhấn Dự đoán để bắt đầu phân tích.</p>
                </div>
            <?php endif; ?>
        </div>

        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:end; gap:12px; margin-bottom: 14px; flex-wrap:wrap;">
                <div>
                    <h3>Lịch sử gần đây</h3>
                    <p class="muted">8 lượt tra cứu gần nhất của bạn.</p>
                </div>
            </div>
            <div class="table-container">
                <table class="table">
                    <thead><tr><th>Input</th><th>Kết quả</th></tr></thead>
                    <tbody>
                    <?php if (empty($histories)): ?><tr><td colspan="2" style="text-align:center;" class="muted">Chưa có dữ liệu</td></tr><?php endif; ?>
                    <?php foreach ($histories as $item): ?>
                        <tr>
                            <td><span class="muted" style="font-size:11px;"><?= e((string) $item['query_type']) ?></span><br><strong><?= e((string) $item['input_text']) ?></strong></td>
                            <td style="text-align:right;"><span class="badge" style="background:rgba(255,255,255,0.05);">Top-<?= e((string) $item['top_k']) ?></span></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="glass-card" style="margin-top:24px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
            <div>
                <h3>Đồ thị tương tác 3D</h3>
                <p class="muted">Trực quan hoá mạng lưới thuốc, bệnh và protein.</p>
            </div>
            <div class="muted">
                <span style="display:inline-block; width:10px; height:10px; background:#60a5fa; border-radius:50%; margin-right:4px;"></span> Thuốc
                <span style="display:inline-block; width:10px; height:10px; background:#f87171; border-radius:50%; margin-left:12px; margin-right:4px;"></span> Bệnh
                <span style="display:inline-block; width:10px; height:10px; background:#fbbf24; border-radius:50%; margin-left:12px; margin-right:4px;"></span> Protein
            </div>
        </div>
        <?php if (empty($graph['nodes'])): ?>
            <div style="height:400px; display:grid; place-items:center; border:1px dashed rgba(148,163,184,0.22); border-radius:18px; color:var(--text-muted); background: rgba(2,6,23,0.34);">Chưa có dữ liệu đồ thị. Vui lòng thực hiện dự đoán.</div>
        <?php else: ?>
            <div id="graph3d"></div>
        <?php endif; ?>
    </div>
</div>

<script>
const graphData = <?= json_encode($graph, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const apiNote = <?= json_encode($resultData['note'] ?? '', JSON_UNESCAPED_UNICODE) ?>;
const phpSuccess = <?= json_encode($success ?? '', JSON_UNESCAPED_UNICODE) ?>;
function showToast(message) { const container = document.getElementById('toast-container'); const toast = document.createElement('div'); const isWarn = message.includes('⚠️'); const isSuccess = message === 'Thành công'; toast.className = `toast ${isWarn ? 'toast-warn' : ''} ${isSuccess ? 'toast-success' : ''}`; toast.innerHTML = `<div class="toast-icon">${isWarn ? '!' : (isSuccess ? '✓' : 'i')}</div><div class="toast-message">${message}</div>`; container.appendChild(toast); setTimeout(() => toast.classList.add('show'), 100); setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 400); }, 5000); }
if (apiNote && (apiNote.includes('⚠️') || apiNote.includes('ℹ️'))) showToast(apiNote); else if (phpSuccess) showToast(phpSuccess);
const container = document.getElementById('graph3d');
if (container && graphData && Array.isArray(graphData.nodes) && graphData.nodes.length > 0) {
    const Graph = ForceGraph3D()(container)
        .graphData(graphData)
        .nodeLabel(n => `${n.name || n.id} (${n.type || 'node'})`)
        .nodeAutoColorBy('type')
        .linkOpacity(0.25)
        .linkWidth(1.1)
        .backgroundColor('rgba(2,6,23,0)');
    Graph.d3Force('charge').strength(-180);
}
</script>
</body>
</html>