import MenuTrigger from "@/components/menu";
import NotificationsTrigger from "@/components/notifications";

export default function DefaultScreen() {
  return (
    <main className="relative z-0 grid h-screen max-h-120 w-full min-w-200 max-w-200 animate-in items-center justify-center bg-linear-to-b from-black to-sky-950 p-4">
      <div className="absolute top-6 left-6 flex items-center gap-x-4">
        <MenuTrigger />
        <NotificationsTrigger />
      </div>
    </main>
  );
}
