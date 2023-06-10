const path = require('path');

const fs = require('fs');

module.exports = {
  resolve: {
    fallback: {
      fs: require.resolve("fs"),
      path: require.resolve("path-browserify")
    }
  }
};
