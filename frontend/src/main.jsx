import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import MonitoringDashboard from "./MonitoringDashboard.jsx";
import "./style.css";

const page = window.location.pathname.replace(/\/+$/, "") === "/monitor"
  ? <MonitoringDashboard />
  : <App />;

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    {page}
  </React.StrictMode>
);
