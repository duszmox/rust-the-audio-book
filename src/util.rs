use chrono::Local;

pub fn now_ts() -> String {
    let now = Local::now();
    now.format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}
