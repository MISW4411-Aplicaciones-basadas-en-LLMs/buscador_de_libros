# Use a lightweight nginx image to serve static content
FROM nginx:alpine

# Copy the static files from the frontend directory to the nginx public folder
COPY ./ /usr/share/nginx/html

# Expose port 80 to the outside world
EXPOSE 80

# The nginx image will automatically start the server
CMD ["nginx", "-g", "daemon off;"]
