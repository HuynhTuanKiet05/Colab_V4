<?php
require_once __DIR__ . '/../app/services/AuthService.php';

if (is_logged_in()) {
    redirect('index.php');
}

$error = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim($_POST['username'] ?? '');
    $password = trim($_POST['password'] ?? '');

    if (AuthService::attemptLogin($username, $password)) {
        redirect('index.php');
    }

    $error = 'Sai tên đăng nhập hoặc mật khẩu.';
}
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Đăng nhập</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="login-wrap">
    <div class="login-box">
        <h1>Đăng nhập hệ thống</h1>
        <p class="muted">Website dự đoán liên kết thuốc - bệnh bằng AI</p>
        <?php if ($error): ?>
            <div class="alert alert-error"><?= e($error) ?></div>
        <?php endif; ?>
        <form method="post">
            <div style="margin-bottom:12px;">
                <label>Tên đăng nhập</label>
                <input class="input" type="text" name="username" required>
            </div>
            <div style="margin-bottom:16px;">
                <label>Mật khẩu</label>
                <input class="input" type="password" name="password" required>
            </div>
            <button class="btn" type="submit">Đăng nhập</button>
        </form>
        <p class="muted" style="margin-top:16px;">Mặc định: admin / password</p>
    </div>
</div>
</body>
</html>
