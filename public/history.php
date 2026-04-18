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
        <div class="brand">Lịch sử tra cứu</div>
        <div class="nav-links">
            <a class="btn btn-secondary" href="index.php">Dashboard</a>
            <a class="btn btn-danger" href="logout.php">Đăng xuất</a>
        </div>
    </div>
    <div class="card">
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
                    <td><?= e((string) $row['query_type']) ?></td>
                    <td><?= e((string) $row['input_text']) ?></td>
                    <td><?= e((string) $row['top_k']) ?></td>
                    <td><?= e((string) $row['status']) ?></td>
                    <td><?= e((string) $row['created_at']) ?></td>
                </tr>
            <?php endforeach; ?>
            </tbody>
        </table>
    </div>
</div>
</body>
</html>
