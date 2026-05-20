# Build stage
FROM node:18-alpine AS build-stage

WORKDIR /app

# VITE_API_BASE is injected at build time. Default points to localhost for
# docker-compose development; override in `docker build --build-arg` for prod.
ARG VITE_API_BASE=http://127.0.0.1:8000
ENV VITE_API_BASE=${VITE_API_BASE}

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# Production stage
FROM nginx:stable-alpine AS production-stage

COPY --from=build-stage /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
