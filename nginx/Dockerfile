# use Nginx to serve static files
FROM nginx:alpine
RUN mkdir -p /usr/share/nginx/html/pdf
COPY --from=builder /app/static/data/pdfs /usr/share/nginx/html/pdfs
EXPOSE 80