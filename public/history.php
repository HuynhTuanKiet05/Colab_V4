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
    <title>Lịch sử tra cứu · AMNTDDA AI</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Lịch sử tra cứu</div>
            <div class="muted">Lưu lại toàn bộ các phiên chẩn đoán đã thực hiện trên hệ thống AMNTDDA AI.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="index.php">Trang chủ</a>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>

    <div class="glass-card">
        <div class="section-header">
            <div>
                <h3>Nhật ký tra cứu</h3>
                <p class="muted">Toàn bộ phiên chẩn đoán của tài khoản hiện tại được hiển thị trong một bảng thống nhất.</p>
            </div>
            <div class="badge badge-drug"><?= count($rows) ?> bản ghi</div>
        </div>

        <div class="table-container">
            <table class="table">
                <thead>
                <tr>
                    <th>ID</th>
                    <th>Kiểu tra cứu</th>
                    <th>Truy vấn</th>
                    <th>Top-K</th>
                    <th>Trạng thái</th>
                    <th>Thời gian</th>
                </tr>
                </thead>
                <tbody>
                <?php if (empty($rows)): ?>
                    <tr><td colspan="6" class="center-empty">Chưa có lịch sử tra cứu.</td></tr>
                <?php endif; ?>
                <?php foreach ($rows as $row): ?>
                    <tr>
                        <td><?= e((string) $row['id']) ?></td>
                        <td><span class="badge badge-neutral"><?= e((string) $row['query_type']) ?></span></td>
                        <td><strong><?= e((string) $row['input_text']) ?></strong></td>
                        <td><span class="badge badge-drug">Top-<?= e((string) $row['top_k']) ?></span></td>
                        <td><span class="badge badge-success">Hoàn tất</span></td>
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
