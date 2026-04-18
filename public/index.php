<?php
require_once __DIR__ . '/../app/services/PredictionService.php';
require_login();

$user = current_user();
$resultData = null;
$error = null;
$queryType = $_POST['query_type'] ?? 'drug_to_disease';
$inputText = trim($_POST['input_text'] ?? '');
$topK = max(1, min(20, (int) ($_POST['top_k'] ?? 10)));
$apiHealthy = PredictionService::isApiHealthy();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if ($inputText === '') {
        $error = 'Vui lòng nhập tên thuốc hoặc bệnh.';
    } elseif (!$apiHealthy) {
        $error = 'Python API chưa chạy. Hãy chạy FastAPI rồi thử lại.';
    } else {
        try {
            $resultData = PredictionService::callPythonApi($queryType, $inputText, $topK);
            PredictionService::saveHistory((int) $user['id'], $queryType, $inputText, $topK, $resultData);
            flash('success', 'Dự đoán thành công và đã lưu lịch sử.');
            $_SESSION['latest_graph'] = $resultData['graph'] ?? ['nodes' => [], 'links' => []];
        } catch (Throwable $e) {
            $error = $e->getMessage();
        }
    }
}

$success = flash('success');
$graph = $resultData['graph'] ?? ($_SESSION['latest_graph'] ?? ['nodes' => [], 'links' => []]);

$historyStmt = db()->prepare('SELECT * FROM prediction_requests WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 10');
$historyStmt->execute(['user_id' => $user['id']]);
$histories = $historyStmt->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <link rel="stylesheet" href="assets/style.css">
    <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
<div class="container">
    <div class="navbar">
        <div class="brand">Drug Disease AI Predictor</div>
        <div class="nav-links">
            <a class="btn btn-secondary" href="history.php">Lịch sử</a>
            <?php if (($user['role'] ?? '') === 'admin'): ?>
                <a class="btn btn-secondary" href="admin.php">Quản trị</a>
            <?php endif; ?>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="hero">
        <h1>Xin chào, <?= e($user['full_name'] ?? $user['username']) ?></h1>
        <p class="muted">Tra cứu thuốc hoặc bệnh, lấy top-k kết quả dự đoán, score và biểu đồ node 3D.</p>
    </div>

    <?php if ($success): ?><div class="alert alert-success"><?= e($success) ?></div><?php endif; ?>
    <?php if (!$apiHealthy): ?><div class="alert alert-error">Python API đang tắt. Hãy chạy <strong>uvicorn main:app --host 127.0.0.1 --port 8000 --reload</strong> trong thư mục <strong>python_api</strong>.</div><?php endif; ?>
    <?php if ($error): ?><div class="alert alert-error"><?= e($error) ?></div><?php endif; ?>

    <div class="grid grid-2">
        <div class="card">
            <h2>Tra cứu dự đoán</h2>
            <form method="post">
                <div class="form-row">
                    <select class="select" name="query_type">
                        <option value="drug_to_disease" <?= $queryType === 'drug_to_disease' ? 'selected' : '' ?>>Nhập thuốc để tìm bệnh tiềm năng</option>
                        <option value="disease_to_drug" <?= $queryType === 'disease_to_drug' ? 'selected' : '' ?>>Nhập bệnh để tìm thuốc tiềm năng</option>
                    </select>
                    <input class="input" type="number" name="top_k" min="1" max="20" value="<?= e((string) $topK) ?>">
                    <button class="btn" type="submit" <?= !$apiHealthy ? 'disabled' : '' ?>>Dự đoán</button>
                </div>
                <div style="margin-top:12px;">
                    <input class="input" type="text" name="input_text" value="<?= e($inputText) ?>" placeholder="Ví dụ: Goserelin hoặc D102100" required>
                </div>
            </form>

            <?php if ($resultData): ?>
                <div style="margin-top:20px;">
                    <h3>Kết quả dự đoán</h3>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Hạng</th>
                                <th>Tên</th>
                                <th>Loại</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                        <?php foreach (($resultData['results'] ?? []) as $index => $item): ?>
                            <tr>
                                <td><?= $index + 1 ?></td>
                                <td><?= e((string) $item['name']) ?></td>
                                <td><span class="badge <?= ($item['type'] ?? '') === 'drug' ? 'badge-drug' : 'badge-disease' ?>"><?= e((string) $item['type']) ?></span></td>
                                <td class="result-score"><?= e(number_format((float) $item['score'], 4)) ?></td>
                            </tr>
                        <?php endforeach; ?>
                        </tbody>
                    </table>
                    <p class="muted" style="margin-top:12px;"><?= e((string) ($resultData['note'] ?? '')) ?></p>
                </div>
            <?php endif; ?>
        </div>

        <div class="card">
            <h2>Lịch sử gần đây</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Kiểu</th>
                        <th>Input</th>
                        <th>Top-k</th>
                        <th>Thời gian</th>
                    </tr>
                </thead>
                <tbody>
                <?php foreach ($histories as $item): ?>
                    <tr>
                        <td><?= e((string) $item['query_type']) ?></td>
                        <td><?= e((string) $item['input_text']) ?></td>
                        <td><?= e((string) $item['top_k']) ?></td>
                        <td><?= e((string) $item['created_at']) ?></td>
                    </tr>
                <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>

    <div class="card" style="margin-top:18px;">
        <h2>Biểu đồ node 3D</h2>
        <p class="muted">Màu xanh: thuốc, đỏ: bệnh, cam: protein.</p>
        <?php if (empty($graph['nodes'])): ?>
            <div class="alert alert-error">Chưa có dữ liệu graph để hiển thị. Hãy chạy dự đoán thành công trước.</div>
        <?php endif; ?>
        <div id="graph3d"></div>
    </div>
    <div class="footer-space"></div>
</div>
<script>
const graphData = <?= json_encode($graph, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?>;
const container = document.getElementById('graph3d');
if (container && graphData && Array.isArray(graphData.nodes)) {
    ForceGraph3D()(container)
        .graphData(graphData)
        .nodeLabel(node => `${node.label} (${node.type})`)
        .nodeAutoColorBy('type')
        .nodeColor(node => node.color || '#ffffff')
        .linkWidth(link => Math.max(1, (link.score || 0.5) * 3))
        .linkDirectionalParticles(1)
        .linkDirectionalParticleSpeed(0.003);
}
</script>
</body>
</html>
