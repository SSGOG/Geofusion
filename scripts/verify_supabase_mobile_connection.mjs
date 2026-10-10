import { readFileSync } from "node:fs";
import { createClient } from "../mobile/node_modules/@supabase/supabase-js/dist/index.mjs";

const values = Object.fromEntries(readFileSync(new URL("../mobile/.env", import.meta.url), "utf8")
  .split(/\r?\n/)
  .filter((line) => line && !line.startsWith("#") && line.includes("="))
  .map((line) => {
    const [key, ...rest] = line.split("=");
    return [key, rest.join("=").trim()];
  }));

const url = values.EXPO_PUBLIC_SUPABASE_URL?.trim()
  .replace(/\/+$/, "")
  .replace(/\/rest\/v1$/i, "");
const key = values.EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
if (!url || !key) throw new Error("Supabase URL or publishable key is missing from mobile/.env.");

const supabase = createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false } });
if (new URL(supabase.rest.url).pathname.replace(/\/+$/, "") !== "/rest/v1") {
  throw new Error("The Supabase project URL does not resolve to the expected REST API path.");
}
console.log("PASS Supabase project URL resolves to its REST API path.");
const { error: signInError } = await supabase.auth.signInAnonymously();
if (signInError) throw signInError;
const { error: queryError } = await supabase.from("mobile_road_events")
  .select("event_id,reporter_id", { head: true, count: "exact" })
  .eq("status", "active");
if (queryError) throw queryError;
console.log("PASS Supabase anonymous sign-in and read policy are available.");
