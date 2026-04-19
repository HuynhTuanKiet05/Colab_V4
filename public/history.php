<?php
require_once __DIR__ . '/../app/bootstrap.php';
require_login();

$stmt = db()->prepare('SELECT pr.*, u.username FROM prediction_requests pr INNER JOIN users u ON u.id = pr.user_id WHERE pr.user_id = :user_id ORDER BY pr.created_at DESC');
$stmt->execute(['user_id' => current_user()['id']]);
$rows = $stmt->fetchAll();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lịch sử tra cứu</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Lịch sử tra cứu</div>
            <div class="muted">Tất cả các truy vấn đã thực hiện trước đây</div>
        </div>
        <div class="nav-links">
            <a class="btn" style="background: rgba(255,255,255,0.06); box-shadow:none;" href="index.php">Dashboard</a>
            <a class="btn" style="background: rgba(244, 63, 94, 0.16); box-shadow:none;" href="logout.php">Đăng xuất</a>
        </div>
    </div>
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:end; margin-bottom:14px; flex-wrap:wrap; gap:12px;">
            <div>
                <h3>Nhật ký tra cứu</h3>
                <p class="muted">Lưu lại toàn bộ lịch sử dự đoán của tài khoản hiện tại.</p>
            </div>
            <div class="badge badge-drug"><?= count($rows) ?> records</div>
        </div>
        <div class="table-container">
            <table class="table">
                <thead>
                <tr>
                    <th>ID</th>
                    <th>Kiểu tra cứu</th>
                    <th>Input</th>
                    <th>Top-k</th>
                    <th>Trạng thái</th>
                    <th>Thời gian</th>
                </tr>
                </thead>
                <tbody>
                <?php foreach ($rows as $row): ?>
                    <tr>
                        <td><?= e((string) $row['id']) ?></td>
                        <td><span class="badge" style="background: rgba(255,255,255,0.05); color:#cbd5e1; border:1px solid rgba(148,163,184,0.16);"><?= e((string) $row['query_type']) ?></span></td>
                        <td><?= e((string) $row['input_text']) ?></td>
                        <td><span class="badge badge-drug">Top-<?= e((string) $row['top_k']) ?></span></td>
                        <td><span class="badge" style="background: rgba(34,197,94,0.12); color:#86efac; border-color: rgba(34,197,94,0.2);">Done</span></td>
                        <td class="muted"><?= e((string) $row['created_at']) ?></td>
                    </tr>
                <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>
</div>
</body>
</html>