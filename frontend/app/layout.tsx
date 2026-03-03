import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PMS Mood Compass",
  description: "Track symptoms and lifestyle factors to estimate daily PMS symptom patterns.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
