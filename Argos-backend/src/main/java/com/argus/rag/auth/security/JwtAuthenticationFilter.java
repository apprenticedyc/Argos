package com.argus.rag.auth.security;

import com.argus.rag.common.api.ApiResponse;
import com.argus.rag.common.exception.BusinessException;
import com.argus.rag.common.security.AuthenticatedUser;
import com.argus.rag.common.security.UserContext;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * JWT 认证过滤器。
 * <p>
 * 从 Authorization 头提取 Bearer token，解析后设置 request attribute 供后续控制器使用。
 * 对 /api/auth/* 路径跳过过滤。
 */
@Slf4j
@Component
// OncePerRequestFilter保证‌一次 HTTP 请求中过滤逻辑只执行一次‌，避免内部转发导致的重复处理
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private static final String BEARER_PREFIX = "Bearer ";
    /** 无需认证的路径 */
    private static final String LOGIN_PATH = "/api/auth/login";
    private static final String REGISTER_PATH = "/api/auth/register";
    private static final String REFRESH_PATH = "/api/auth/refresh";
    private static final String LOGOUT_PATH = "/api/auth/logout";

    private final AccessTokenService accessTokenService;
    private final ObjectMapper objectMapper;

    public JwtAuthenticationFilter(AccessTokenService accessTokenService, ObjectMapper objectMapper) {
        this.accessTokenService = accessTokenService;
        this.objectMapper = objectMapper;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        // 取 Authorization 头，没有则放行（由后续权限校验处理）
        String authorization = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (authorization == null) {
            log.debug("请求无 Authorization header: path={}", request.getRequestURI());
            filterChain.doFilter(request, response);
            return;
        }

        // 提取 token：兼容 "Bearer xxx" 和裸 token 两种格式
        String accessToken;
        if (authorization.startsWith(BEARER_PREFIX)) {
            accessToken = authorization.substring(BEARER_PREFIX.length()).trim();
        } else {
            accessToken = authorization.trim();
        }
        if (accessToken.isEmpty()) {
            writeUnauthorized(response, "access token 非法或已过期");
            return;
        }

        try {
            // 解析 JWT，验证签名和过期时间
            AccessTokenService.AccessTokenClaims claims = accessTokenService.parse(accessToken);
            log.debug("JWT 认证成功: userId={}, path={}", claims.userId(), request.getRequestURI());
            // 认证通过，将用户信息写入 ThreadLocal 供后续使用
            UserContext.set(new AuthenticatedUser(claims.userId(), claims.userCode(), claims.displayName(), claims.systemRole(), claims.mustChangePassword()));
            filterChain.doFilter(request, response);
        } catch (BusinessException exception) {
            // 签名无效或 token 过期，返回 401
            log.debug("JWT 认证失败: {} path={}", exception.getMessage(), request.getRequestURI());
            writeUnauthorized(response, exception.getMessage());
        } finally {
            // 清理 ThreadLocal，防止线程复用导致数据泄漏
            UserContext.clear();
        }
    }

    /** 认证白名单路径跳过过滤 */
    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String requestUri = request.getRequestURI();
        return LOGIN_PATH.equals(requestUri) || REGISTER_PATH.equals(requestUri) || REFRESH_PATH.equals(requestUri) || LOGOUT_PATH.equals(requestUri);
    }

    private void writeUnauthorized(HttpServletResponse response, String message) throws IOException {
        // 设置 401 状态码
        response.setStatus(HttpStatus.UNAUTHORIZED.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        objectMapper.writeValue(response.getWriter(), new ApiResponse<>(false, null, message));
    }

}
