import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import MonitoringDashboard from "./MonitoringDashboard.jsx";
import CatalogPage from "./CatalogPage.jsx";
import "./style.css";

const path = window.location.pathname.replace(/\/+$/, "");
const page = path === "/monitor" ? <MonitoringDashboard />
  : path === "/catalogo" ? <CatalogPage />
  : <App />;

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    {page}
  </React.StrictMode>
);
