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
        <div style="margin-bottom: 20px;">
            <div class="brand" style="font-size: 28px; display: inline-block; margin-bottom: 10px;">AMNTDDA</div>
            <h1>Đăng nhập hệ thống</h1>
            <p class="muted">Nền tảng dự đoán liên kết thuốc - bệnh bằng AI HGT</p>
        </div>
        <?php if ($error): ?>
            <div class="alert alert-error"><?= e($error) ?></div>
        <?php endif; ?>
        <form method="post">
            <div style="margin-bottom:14px;">
                <label>Tên đăng nhập</label>
                <input class="input" type="text" name="username" placeholder="Nhập username" required>
            </div>
            <div style="margin-bottom:18px;">
                <label>Mật khẩu</label>
                <input class="input" type="password" name="password" placeholder="Nhập mật khẩu" required>
            </div>
            <button class="btn" type="submit" style="width:100%;">Đăng nhập</button>
        </form>
        <div style="margin-top:18px; padding: 14px 16px; border-radius: 16px; background: rgba(96, 165, 250, 0.08); border: 1px solid rgba(96, 165, 250, 0.16);">
            <p class="muted" style="margin:0;">Mặc định: <strong>admin / password</strong></p>
        </div>
    </div>
</div>
</body>
</html>