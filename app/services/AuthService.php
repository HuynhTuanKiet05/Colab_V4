<?php

declare(strict_types=1);

require_once __DIR__ . '/../bootstrap.php';

class AuthService
{
    public static function attemptLogin(string $username, string $password): bool
    {
        $sql = 'SELECT * FROM users WHERE username = :username AND status = :status LIMIT 1';
        $stmt = db()->prepare($sql);
        $stmt->execute([
            'username' => $username,
            'status' => 'active',
        ]);

        $user = $stmt->fetch();

        if (!$user || !password_verify($password, $user['password_hash'])) {
            return false;
        }

        unset($user['password_hash']);
        $_SESSION['user'] = $user;

        $update = db()->prepare('UPDATE users SET last_login_at = NOW() WHERE id = :id');
        $update->execute(['id' => $user['id']]);

        return true;
    }

    public static function logout(): void
    {
        unset($_SESSION['user']);
    }
}
