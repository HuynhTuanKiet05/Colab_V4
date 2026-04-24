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
    <title>Lich su tra cuu</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <div class="navbar">
        <div>
            <div class="brand">Lich su tra cuu</div>
            <div class="muted">Luu lai toan bo cac truy van du doan da thuc hien truoc day.</div>
        </div>
        <div class="nav-links">
            <a class="btn btn-ghost" href="index.php">Dashboard</a>
            <a class="btn btn-ghost" href="compare_models.php">So sanh 2 mo hinh</a>
            <a class="btn btn-danger" href="logout.php">Dang xuat</a>
        </div>
    </div>

    <div class="glass-card">
        <div class="section-header">
            <div>
                <h3>Nhat ky tra cuu</h3>
                <p class="muted">Toan bo lich su du doan cua tai khoan hien tai duoc hien thi trong mot bang thong nhat.</p>
            </div>
            <div class="badge badge-drug"><?= count($rows) ?> records</div>
        </div>

        <div class="table-container">
            <table class="table">
                <thead>
                <tr>
                    <th>ID</th>
                    <th>Kieu tra cuu</th>
                    <th>Input</th>
                    <th>Top-k</th>
                    <th>Trang thai</th>
                    <th>Thoi gian</th>
                </tr>
                </thead>
                <tbody>
                <?php if (empty($rows)): ?>
                    <tr><td colspan="6" class="center-empty">Chua co lich su tra cuu.</td></tr>
                <?php endif; ?>
                <?php foreach ($rows as $row): ?>
                    <tr>
                        <td><?= e((string) $row['id']) ?></td>
                        <td><span class="badge badge-neutral"><?= e((string) $row['query_type']) ?></span></td>
                        <td><strong><?= e((string) $row['input_text']) ?></strong></td>
                        <td><span class="badge badge-drug">Top-<?= e((string) $row['top_k']) ?></span></td>
                        <td><span class="badge badge-success">Done</span></td>
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
