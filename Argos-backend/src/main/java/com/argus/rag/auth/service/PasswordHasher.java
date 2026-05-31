package com.argus.rag.auth.service;

import com.argus.rag.common.exception.BusinessException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;

/**
 * 基于 BCrypt 的密码哈希工具。
 */
@Component
public class PasswordHasher {

    /** BCrypt算法原生最大支持输入字节数 */
    private static final int BCRYPT_MAX_INPUT_BYTES = 72;

    private final BCryptPasswordEncoder encoder = new BCryptPasswordEncoder();

    /**
     * 使用 BCrypt 对明文密码哈希加密
     * @param rawPassword 原始明文
     * @return BCrypt哈希密文
     */
    public String hash(String rawPassword) {
        validateInputLength(rawPassword);
        return encoder.encode(rawPassword);
    }

    /**
     * 校验明文与哈希密文是否匹配
     * @param raw 原始明文
     * @param hash 存储的哈希密文
     * @return 匹配返回true
     */
    public boolean matches(String raw, String hash) {
        validateInputLength(raw);
        return encoder.matches(raw, hash);
    }

    /**
     * 校验输入字节长度，超出72字节抛出异常
     */
    private void validateInputLength(String rawValue) {
        if (rawValue == null || rawValue.getBytes(StandardCharsets.UTF_8).length > BCRYPT_MAX_INPUT_BYTES) {
            throw new BusinessException("密码长度超过安全上限，请控制在 72 字节以内");
        }
    }
}